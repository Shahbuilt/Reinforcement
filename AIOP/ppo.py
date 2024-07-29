import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
import pandas as pd
from tensorflow.keras import backend as K

class ReplayBuffer:
    def __init__(self, agentIndex, agent_lookback, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.agent_lookback = agent_lookback
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.log_probs = []

    def store_episode(self, states, actions, rewards, dones, next_states, log_probs):
        for state, action, reward, done, next_state, log_prob in zip(states, actions, rewards, dones, next_states, log_probs):
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
            self.next_states.append(next_state)
            self.log_probs.append(log_prob)
        if len(self.states) > self.capacity:
            self.states = self.states[-self.capacity:]
            self.actions = self.actions[-self.capacity:]
            self.rewards = self.rewards[-self.capacity:]
            self.dones = self.dones[-self.capacity:]
            self.next_states = self.next_states[-self.capacity:]
            self.log_probs = self.log_probs[-self.capacity:]            

    # def get_entire_episode(self):
    #     return (
    #         np.array(self.states),
    #         np.array(self.actions),
    #         np.array(self.rewards),
    #         np.array(self.dones),
    #         np.array(self.next_states),
    #         np.array(self.log_probs)
    #     )
    def sample_batch(self):
        idx = np.random.randint(0, len(self.states), size=self.batch_size)
        return (
            np.array(self.states)[idx],
            np.array(self.actions)[idx],
            np.array(self.rewards)[idx],
            np.array(self.dones)[idx],
            np.array(self.next_states)[idx],
            np.array(self.log_probs)[idx]
        )

    def saveEpisodes(self, path):
        data = {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones,
            'next_states': self.next_states,
            'log_probs': self.log_probs
        }
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    def get_min_max(self):
        data = {
            'states': np.array(self.states).reshape(-1, np.array(self.states).shape[-1]),
            'next_states': np.array(self.next_states).reshape(-1, np.array(self.next_states).shape[-1])
        }
        df = pd.DataFrame(data)
        buff_min = df.min()
        buff_max = df.max()
        buff_mean = df.mean()
        buff_std = df.std()
        return buff_min, buff_max, buff_mean, buff_std

class PPOAgent:
    def __init__(self, agentIndex, MVindex, agent_lookback, gamma, lambda_, clip_epsilon, entropy_coef, critic_coef, max_step, training_scanrate, execution_scanrate, learning_rate, std=0.3, action_space = 1):
        self.agentIndex = agentIndex
        self.MVindex = MVindex
        self.agent_lookback = agent_lookback
        self.gamma = gamma
        self.action_space = action_space
        self.std = std
        self.log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        self.lambda_ = lambda_
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.max_step = max_step
        self.training_scanrate = training_scanrate
        self.execution_scanrate = execution_scanrate
        self.learning_rate = learning_rate

        self.actor = None
        self.critic = None
        self.num_actions = len(agentIndex)

    def build_actor(self, input_dims, lr):
        inputs = Input(shape=(self.agent_lookback, input_dims))
        # self.action_space = 1
        x = Dense(128, activation='relu')(inputs)
        # x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)  # Changed to tanh for continuous action space
        self.actor = Model(inputs, outputs)
        self.actor.compile(optimizer=Adam(learning_rate=0.0003), loss=self.ppo_loss)
        # self.actor.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    def build_critic(self, input_dims, lr):
        inputs = Input(shape=(self.agent_lookback, input_dims))
        x = Dense(128, activation='relu')(inputs)
        # x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(1, activation='tanh')(x)
        self.critic = Model(inputs, outputs)
        self.critic.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')


    # def ppo_loss(self, y_true, y_pred):
    #     # Extracting advantages and old predictions
    #     advantages = y_true[:, 0]
    #     old_predictions = y_true[:, 1:]
        
    #     # Clip values to avoid log(0)
    #     old_predictions = tf.clip_by_value(old_predictions, 1e-10, 1.0)
    #     y_pred = tf.clip_by_value(y_pred, 1e-10, 1.0)
    #     old_predictions =tf.squeeze(old_predictions, axis=-1)
    #     y_pred = tf.squeeze(y_pred, axis=-1)
        
    #     # Calculate probability ratios
    #     prob_ratios = tf.exp(tf.math.log(y_pred) - tf.math.log(old_predictions))
    #     clipped_prob_ratios = tf.clip_by_value(prob_ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        
    #     # Compute surrogate losses
    #     surrogate1 = prob_ratios * advantages
    #     surrogate2 = clipped_prob_ratios * advantages
 
        
    #     # Compute actor and entropy losses
    #     actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
    #     entropy_loss = self.entropy_coef * tf.reduce_mean(-y_pred * tf.math.log(y_pred))
    #     # Return total loss
    #     loss = actor_loss - entropy_loss
    #     return loss

    def ppo_loss(self, y_true, y_pred):
        # Extracting advantages and old predictions
        advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space]
        LOSS_CLIPPING = 0.2
        logp = self.gaussian_likelihood_loss(actions, y_pred)
        ratio = K.exp(logp - logp_old_ph)
        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING)*advantages, (1.0 - LOSS_CLIPPING)*advantages) # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss
        
        # Clip values to avoid log(0)
        old_predictions = tf.clip_by_value(old_predictions, 1e-10, 1.0)
        y_pred = tf.clip_by_value(y_pred, 1e-10, 1.0)
        old_predictions =tf.squeeze(old_predictions, axis=-1)
        y_pred = tf.squeeze(y_pred, axis=-1)
        
        # Calculate probability ratios
        prob_ratios = tf.exp(tf.math.log(y_pred) - tf.math.log(old_predictions))
        clipped_prob_ratios = tf.clip_by_value(prob_ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        
        # Compute surrogate losses
        surrogate1 = prob_ratios * advantages
        surrogate2 = clipped_prob_ratios * advantages
 
        
        # Compute actor and entropy losses
        actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
        entropy_loss = self.entropy_coef * tf.reduce_mean(-y_pred * tf.math.log(y_pred))
        # Return total loss
        loss = actor_loss - entropy_loss
        return loss

    def gaussian_likelihood_loss(self, actions, pred): # for keras custom loss
        log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        pre_sum = -0.5 * (((actions-pred)/(K.exp(log_std)+1e-8))**2 + 2*log_std + K.log(2*np.pi))
        return K.sum(pre_sum, axis=1)
    
    def select_action(self, state):
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        pred = self.actor.predict(state, verbose = 1)
        action = pred + np.random.normal(0, self.std, size=pred.shape)
        action = np.clip(action, 0, 1.0)  # Ensure actions are within valid range
        # print("action is: ",action)
        # log_prob = -0.5 * ((action - pred) ** 2 / (self.std ** 2) + np.log(2 * np.pi * self.std ** 2 + 1e-10))  # Avoid log(0)
        log_prob = -0.5 * ((action - pred) / (np.exp(self.log_std)+1e-8))**2 + 2*self.log_std + np.log(2*np.pi)
        return action.flatten(), np.sum(log_prob,axis =1)

    def learn(self, states, actions, rewards, dones, next_states, old_log_probs):
        # states, actions, rewards, dones, next_states, old_log_probs = replay_buffer.get_entire_episode()
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        rewards = np.squeeze(rewards)
        dones = np.array(dones)
        next_states = np.array(next_states)
        old_log_probs = np.array(old_log_probs)
        old_log_probs = np.squeeze(old_log_probs)
        values = self.critic.predict(states, verbose = 0)
        values = np.squeeze(values)
        next_values = self.critic.predict(next_states, verbose = 0)
        next_values = np.squeeze(next_values)
        advantages, returns = self.compute_advantages(rewards, dones, values, next_values)
        actions = np.reshape(actions, (-1, 1))
        old_log_probs = np.reshape(old_log_probs, (-1, 1))
        advantages = np.reshape(advantages, (-1, 1))
        y_true = np.hstack([advantages, actions, old_log_probs])
        print("states:", states)
        print("y true: ",y_true)
        print("return: ", returns)
        print("reward: ", rewards)
        self.actor.fit(states, y_true, epochs= 10, verbose=1)
        self.critic.fit(states, returns, epochs=10, verbose=1)

    def compute_advantages(self, rewards, dones, values, next_values, normalize = True):
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = np.zeros_like(deltas)
        advantage = 0.0
        for t in reversed(range(len(deltas)-1)):
            advantage = deltas[t] + self.gamma * self.lambda_ * (1 - dones[t]) * deltas[t+1]
            # advantage = deltas[t] + self.gamma * self.lambda_ * (1 - dones[t]) * advantage
            advantages[t] = advantage
        returns = advantages + values
        if normalize:
            advantages = (advantages - advantages.mean())/(advantages.std()+1e-8)
        return advantages, returns

    def save_policy(self, path):
        self.actor.save(path + '_actor.h5')
        self.critic.save(path + '_critic.h5')
        print("model saved")
