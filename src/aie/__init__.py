from ai_economist.foundation.base.base_env import scenario_registry
from ai_economist.foundation.base.base_component import component_registry





from aie.environments.planner_like_agents_env import PlannerLikeAgentsEnv
from aie.environments.planner_like_agents_env import PlannerLikeAgentsEnvTeams

from aie.component_add.add_component import ResourceRedistribution





scenario_registry.add(PlannerLikeAgentsEnv)
scenario_registry.add(PlannerLikeAgentsEnvTeams)

component_registry.add(ResourceRedistribution)
