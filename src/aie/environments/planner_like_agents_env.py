import numpy as np
from ai_economist.foundation.base.base_env import scenario_registry
from ai_economist.foundation.scenarios.simple_wood_and_stone.layout_from_file import LayoutFromFile
from ai_economist.foundation.scenarios.utils import rewards


@scenario_registry.add
class PlannerLikeAgentsEnv(LayoutFromFile):
    name = "PlannerLikeAgentsEnv"

    def get_current_optimization_metrics(self):
        curr_optimization_metric = {}
        # (for agents)
        agent_reward = rewards.coin_eq_times_productivity(
            coin_endowments=np.array([agent.total_endowment("Coin") for agent in self.world.agents]),
            equality_weight=1 - self.mixing_weight_gini_vs_coin,
        )
        for agent in self.world.agents:
            # scale reward to be close to original (utility rewards)
            curr_optimization_metric[agent.idx] = agent_reward / 10.

        # (for the planner)
        if self.planner_reward_type == "coin_eq_times_productivity":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.coin_eq_times_productivity(
                coin_endowments=np.array(
                    [agent.total_endowment("Coin") for agent in self.world.agents]
                ),
                equality_weight=1 - self.mixing_weight_gini_vs_coin,
            )
        elif self.planner_reward_type == "inv_income_weighted_coin_endowments":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.inv_income_weighted_coin_endowments(
                coin_endowments=np.array(
                    [agent.total_endowment("Coin") for agent in self.world.agents]
                ),
            )
        elif self.planner_reward_type == "inv_income_weighted_utility":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.inv_income_weighted_utility(
                coin_endowments=np.array(
                    [agent.total_endowment("Coin") for agent in self.world.agents]
                ),
                utilities=np.array(
                    [curr_optimization_metric[agent.idx] for agent in self.world.agents]
                ),
            )
        else:
            print("No valid planner reward selected!")
            raise NotImplementedError
        return curr_optimization_metric



@scenario_registry.add
class PlannerLikeAgentsEnvTeams(LayoutFromFile):
    name = "PlannerLikeAgentsEnvTeams"




    def get_current_optimization_metrics(self):
        curr_optimization_metric = {}

        team_membership = {0:[0,1], #team: indices
                            1:[2,3]}

        team_membership_agent = {0:0, #team: indices
                                1:0,
                                2:1,
                                3:1}

        coin_arr = []
        for agent in self.world.agents:
            coin_arr.append(agent.total_endowment("Coin"))

        team_reward = []
        for t in range(len(team_membership.keys())):
            team_tot_coin = []
            for idx in team_membership[t]:
                team_tot_coin.append(coin_arr[idx])
            # (for agents)
            team_reward.append(rewards.coin_eq_times_productivity(
                coin_endowments=np.array(team_tot_coin),
                equality_weight=1 - self.mixing_weight_gini_vs_coin,
            ))
        for agent_idx, agent in enumerate(self.world.agents):
            # scale reward to be close to original (utility rewards)
            # curr_optimization_metric[agent.idx] = agent_reward / 10.
            curr_optimization_metric[agent.idx] = team_reward[team_membership_agent[agent_idx]] / 10.

        population_team_coin = [team_reward[team_membership_agent[agent_idx]] for agent_idx, agent in enumerate(self.world.agents)]

        # (for the planner)
        if self.planner_reward_type == "coin_eq_times_productivity":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.coin_eq_times_productivity(
                coin_endowments=np.array(
                    population_team_coin
                ),
                equality_weight=1 - self.mixing_weight_gini_vs_coin,
            )
        elif self.planner_reward_type == "inv_income_weighted_coin_endowments":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.inv_income_weighted_coin_endowments(
                coin_endowments=np.array(
                    population_team_coin
                ),
            )
        elif self.planner_reward_type == "inv_income_weighted_utility":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.inv_income_weighted_utility(
                coin_endowments=np.array(
                    population_team_coin
                ),
                utilities=np.array(
                    [curr_optimization_metric[agent.idx] for agent in self.world.agents]
                ),
            )
        else:
            print("No valid planner reward selected!")
            raise NotImplementedError
        return curr_optimization_metric
