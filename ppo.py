import os

import ray
from ray import air, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF1Policy
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import train_one_step
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import (
    MultiAgentReplayBuffer,
)
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.policy.sample_batch import MultiAgentBatch, concat_samples
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_TARGET_UPDATES,
    LAST_TARGET_UPDATE_TS,
)
from ray.rllib.utils.sgd import standardized
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.typing import ResultDict
from ray.tune.registry import register_env

class MyAlgo(Algorithm):

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        # Generate common experiences, collect batch for PPO, store every (DQN) batch
        # into replay buffer.
        ppo_batches = []
        num_env_steps = 0

        # PPO batch size fixed at 200.
        # TODO: Use `max_env_steps=200` option of synchronous_parallel_sample instead.
        while num_env_steps < 200:
            ma_batches = synchronous_parallel_sample(
                worker_set=self.workers, concat=False
            )
            # Loop through ma-batches (which were collected in parallel).
            for ma_batch in ma_batches:
                # Update sampled counters.
                self._counters[NUM_ENV_STEPS_SAMPLED] += ma_batch.count
                self._counters[NUM_AGENT_STEPS_SAMPLED] += ma_batch.agent_steps()
                ppo_batch = ma_batch.policy_batches.pop("ppo_policy")
                # Add collected batches (only for DQN policy) to replay buffer.
                self.local_replay_buffer.add(ma_batch)

                ppo_batches.append(ppo_batch)
                num_env_steps += ppo_batch.count

        # DQN sub-flow.
        dqn_train_results = {}
        # Start updating DQN policy once we have some samples in the buffer.
        if self._counters[NUM_ENV_STEPS_SAMPLED] > 1000:
            # Update DQN policy n times while updating PPO policy once.
            for _ in range(10):
                dqn_train_batch = self.local_replay_buffer.sample(num_items=64)
                dqn_train_results = train_one_step(
                    self, dqn_train_batch, ["dqn_policy"]
                )
                self._counters[
                    "agent_steps_trained_DQN"
                ] += dqn_train_batch.agent_steps()
                print(
                    "DQN policy learning on samples from",
                    "agent steps trained",
                    dqn_train_batch.agent_steps(),
                )
        # Update DQN's target net every n train steps (determined by the DQN config).
        if (
            self._counters["agent_steps_trained_DQN"]
            - self._counters[LAST_TARGET_UPDATE_TS]
            >= self.get_policy("dqn_policy").config["target_network_update_freq"]
        ):
            self.workers.local_worker().get_policy("dqn_policy").update_target()
            self._counters[NUM_TARGET_UPDATES] += 1
            self._counters[LAST_TARGET_UPDATE_TS] = self._counters[
                "agent_steps_trained_DQN"
            ]

        # PPO sub-flow.
        ppo_train_batch = concat_samples(ppo_batches)
        self._counters["agent_steps_trained_PPO"] += ppo_train_batch.agent_steps()
        # Standardize advantages.
        ppo_train_batch[Postprocessing.ADVANTAGES] = standardized(
            ppo_train_batch[Postprocessing.ADVANTAGES]
        )
        print(
            "PPO policy learning on samples from",
            "agent steps trained",
            ppo_train_batch.agent_steps(),
        )
        ppo_train_batch = MultiAgentBatch(
            {"ppo_policy": ppo_train_batch}, ppo_train_batch.count
        )
        ppo_train_results = train_one_step(self, ppo_train_batch, ["ppo_policy"])

        # Combine results for PPO and DQN into one results dict.
        results = dict(ppo_train_results, **dqn_train_results)
        return results
    

for _ in range(1000):
    print(algo.train())