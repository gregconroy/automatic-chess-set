import numpy as np

class GaussianHMM:
    def __init__(self, n_states, n_components):
        self.n_states = n_states
        self.n_components = n_components

        # Initialize the transition probabilities, initial probabilities, means, and covariances
        self.transition_probabilities = np.random.rand(n_states, n_states)
        self.initial_probabilities = np.random.rand(n_states)
        self.means = np.random.rand(n_states, n_components)
        self.covariances = np.array([np.eye(n_components) for _ in range(n_states)])

        # Normalize the probabilities
        self.initial_probabilities /= np.sum(self.initial_probabilities)
        self.transition_probabilities /= np.sum(self.transition_probabilities, axis=1, keepdims=True)

    def forward(self, observations):
        """Implementation of the forward algorithm in log space."""
        n_obs = observations.shape[0]
        alpha = np.zeros((n_obs, self.n_states))

        # Initial probabilities
        for s in range(self.n_states):
            alpha[0, s] = np.log(self.initial_probabilities[s]) + self._log_gaussian_density(observations[0], self.means[s], self.covariances[s])

        # Forward calculation
        for t in range(1, n_obs):
            for s in range(self.n_states):
                # Use log-sum-exp for stability
                alpha[t, s] = self.log_sum_exp(alpha[t - 1] + np.log(self.transition_probabilities[:, s]) + self._log_gaussian_density(observations[t], self.means[s], self.covariances[s]))

        return alpha

    def backward(self, observations):
        """Implementation of the backward algorithm in log space."""
        n_obs = observations.shape[0]
        beta = np.zeros((n_obs, self.n_states))

        # Setting beta(T) = 0 in log space (log(1) = 0)
        beta[-1] = 0

        # Backward calculation
        for t in range(n_obs - 2, -1, -1):
            for s in range(self.n_states):
                beta[t, s] = self.log_sum_exp(np.log(self.transition_probabilities[s, :]) + 
                                               self._log_gaussian_density(observations[t + 1], self.means[s], self.covariances[s]) + 
                                               beta[t + 1])

        return beta

    def log_sum_exp(self, log_probs):
        """Compute the log sum exp in a numerically stable way."""
        max_log_prob = np.max(log_probs)
        return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))

    def train(self, observations, n_iter=100):
        """Train the HMM using the EM algorithm."""
        n_obs = observations.shape[0]

        for _ in range(n_iter):
            # E-step
            alpha = self.forward(observations)
            beta = self.backward(observations)

            # Compute the expected counts
            xi = np.zeros((self.n_states, self.n_states, n_obs - 1))
            gamma = np.zeros((self.n_states, n_obs))

            for t in range(n_obs - 1):
                # Calculate the log-sum-exp for normalization
                log_sum_exp_val = self.log_sum_exp(alpha[t, :] + beta[t, :])

                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[i, j, t] = (alpha[t, i] + np.log(self.transition_probabilities[i, j]) +
                                       beta[t + 1, j] + self._log_gaussian_density(observations[t + 1], self.means[j], self.covariances[j])) - log_sum_exp_val

                # Update gamma
                gamma[:, t] = self.log_sum_exp(xi[:, :, t])  # Normalize xi to get gamma

            # Correct last gamma
            gamma[:, -1] = alpha[-1]  # Use alpha for the last time step
            sum_gamma_last = np.exp(gamma[:, -1]).sum()  # Sum in the original space
            if sum_gamma_last > 0:
                gamma[:, -1] -= np.log(sum_gamma_last)  # Normalize in log space

            # M-step
            self.initial_probabilities = np.exp(gamma[:, 0] - self.log_sum_exp(gamma[:, 0])) if np.sum(np.exp(gamma[:, 0])) > 0 else self.initial_probabilities

            for i in range(self.n_states):
                # Update means
                total_gamma = np.exp(gamma[i, :]).sum()
                if total_gamma > 0:
                    self.means[i] = np.sum(np.exp(gamma[i, :, np.newaxis]) * observations, axis=0) / total_gamma
                else:
                    print(f"Warning: Total gamma for state {i} is zero. Keeping previous mean.")

                # Update covariance
                diff = observations - self.means[i]
                if total_gamma > 0:
                    # Use a more direct calculation that ensures proper dimensions
                    weighted_diff = np.exp(gamma[i, :, np.newaxis]) * diff  # Shape: (N, D)
                    self.covariances[i] = np.dot(weighted_diff.T, diff) / total_gamma  # Shape: (D, D)
                else:
                    # print(f"Warning: Total gamma for state {i} is zero. Keeping previous covariance.")
                    pass


            # Update transition probabilities
            transition_denom = np.exp(gamma[:, :-1]).sum(axis=1, keepdims=True)
            transition_denom = np.where(transition_denom == 0, 1, transition_denom)  # Avoid zero division
            self.transition_probabilities = np.sum(np.exp(xi), axis=2) / transition_denom

    def likelihood(self, observations):
        """Calculate the likelihood of the observations."""
        n_obs = observations.shape[0]

        # Initialize a likelihood array
        likelihoods = np.zeros((n_obs, self.n_states))

        # Compute likelihood for each observation at each state
        for t in range(n_obs):
            for s in range(self.n_states):
                likelihoods[t, s] = self._log_gaussian_density(observations[t], self.means[s], self.covariances[s])

        # Combine the likelihoods across states for each observation (sum of logs)
        epsilon = 1e-256  # A small constant
        likelihoods = np.clip(likelihoods, epsilon, None)  # Ensures no likelihoods are less than epsilon
        total_likelihood = self.log_sum_exp(likelihoods)

        return total_likelihood

    def predict(self, observations):
        """Implement the prediction logic, e.g., using the Viterbi algorithm."""
        pass  # Implement the prediction logic here

    def _log_gaussian_density(self, x, mean, covariance):
        """Calculate the log of the Gaussian probability density function."""
        d = len(mean)  # Number of dimensions
        # Calculate the determinant of the covariance matrix
        det_covariance = np.linalg.det(covariance)
        if det_covariance == 0:
            return -np.inf  # Avoid division by zero if the covariance is singular

        # Calculate the exponent term
        exponent = -0.5 * (x - mean).T @ np.linalg.inv(covariance) @ (x - mean)

        # Compute the log of the Gaussian density
        log_density = (-0.5 * d * np.log(2 * np.pi) - 0.5 * np.log(det_covariance) + exponent)
        return log_density
