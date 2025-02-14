from pathlib import Path

import matplotlib.pyplot as plt
import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from tqdm import tqdm

from aie import plotting
from aie.aie_env import AIEEnv
from aie.env_conf import ENV_CONF_TEAMS_MACH
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

# ckpt_path = './models/teams_mach/checkpoint_007850/checkpoint-7850' # mach non-RS

ckpt_path = './models/RS_teams_mach/checkpoint_014150/checkpoint-14150' # mach RS

trainer.restore(str(ckpt_path))

# %%
env = AIEEnv(ENV_CONF_TEAMS_MACH, force_dense_logging=True)
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
plotting.breakdown(env.env.previous_episode_dense_log)
plt.show()

# %%
env.env.scenario_metrics()
