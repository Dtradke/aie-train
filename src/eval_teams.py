import matplotlib.pyplot as plt
import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from tqdm import tqdm
import os

from aie import plotting
from aie.aie_env import AIEEnv
from aie.env_conf import ENV_CONF_COMM_TEAMS, ENV_CONF_TEAMS_COMM_RS, ENV_CONF_TEAMS_DEFAULT_RS
from rl.conf import BASE_PPO_CONF, OUT_DIR
from rl.models.tf.fcnet import FCNet

# %%
ray.init()
ModelCatalog.register_custom_model("my_model", FCNet)

# %%
trainer = ppo.PPOTrainer(config={
    **BASE_PPO_CONF,
    "num_workers": 0,
})



# ckpt_path = OUT_DIR / 'PPO_AIEEnv_2021-06-21_11-29-52brgz1h6u/checkpoint_007934/checkpoint-7934'


# ckpt_path = './models/teams_communism/checkpoint_007934/checkpoint-7934' # communism non-RS
ckpt_path = './models/RS_teams_communism/checkpoint_007920/checkpoint-7920' # communism RS MAKE GRAPHS

# ckpt_path = './models/RS_teams_free_market/checkpoint_005885/checkpoint-5885' # free market (default) RS MAKE GRAPHS


trainer.restore(str(ckpt_path))

# %%
# env = AIEEnv(ENV_CONF_TEAMS, force_dense_logging=True) # free market non-RS
# env = AIEEnv(ENV_CONF_COMM_TEAMS, force_dense_logging=True) # communism non-RS

env = AIEEnv(ENV_CONF_TEAMS_COMM_RS, force_dense_logging=True) # communism RS

obs = env.reset()

for t in tqdm(range(1000)):
    results = {
        k: trainer.compute_action(
            v,
            policy_id='learned',
            explore=False,
        )
        for k, v in obs.items()
    }
    obs, reward, done, info = env.step(results)

# %%
# print("about to plot")
plotting.breakdown(env.env.previous_episode_dense_log)
plt.show()
# exit()
# fname = './figs/eval_teams.png'
# plt.savefig(fname,bbox_inches='tight', dpi=300)
# plt.close()

# %%
env.env.scenario_metrics()
