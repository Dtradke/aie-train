

from copy import deepcopy

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls



from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)
from ai_economist.foundation.components.utils import (
    annealed_tax_limit,
    annealed_tax_mask,
)

from ai_economist.foundation.scenarios.utils import rewards



class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal')) #tanh # 2 TEAMS = relu
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal') #linear # 2 TEAMS = softmax

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output



class DQN:
    def __init__(self, type):
        self.nactions = 3 #len(self.world.agents) // 2

        # actions:
        # 0: [0,1], [2,3]
        # 1: [0,2], [1,3]
        # 2: [0,3], [1,2]


        self.batch_size = 32
        self.optimizer = tf.optimizers.Adam(0.0001)
        self.gamma = 0.99

        self.max_experiences = 100000
        self.min_experiences = 32

        self.obs = 4
        hidden_units = [200,200]

        self.model = MyModel(self.obs, hidden_units, self.nactions)

        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def get_action(self, states, epsilon):
        try:
            q_vals = self.predict(np.atleast_2d(states))[0].numpy()
            print(">>>>>> qvals: ", q_vals)
        except:
            q_vals = self.predict(np.atleast_2d(states))
            epsilon = 1
            print(">>>>>> random action, ", q_vals)
        if np.random.random() < epsilon:
            return np.random.choice(self.nactions)
        else:
            return np.argmax(q_vals)


    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next) # adds expected value of future reward whenever not "done"

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.nactions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables

        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())



# redistribute inventory within a team
@component_registry.add
class PlannerTeammaker(BaseComponent):
    """Creates teams for mobile agents.


    """

    name = "PlannerTeammaker"
    required_entities = []
    agent_subclasses = ["BasicMobileAgent"]


    def __init__(
        self,
        *base_component_args,
        period=1,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)


        self._trainPlanner = DQN("train")
        self._targetPlanner = DQN("train")

        self._losses = []
        self._iter = 0

        self._copy_step = 200

        self.eps = 0.5 #TODO: THIS WILL CHANGE

        # self.teams = teams
        # self.team_membership = {0:self.teams[0], #team: indices
        #                     1:self.teams[1]}
        #
        # self.team_membership_agent = {0:0, #indices: team
        #                         1:0,
        #                         2:1,
        #                         3:1}

    """
    Required methods for implementing components
    --------------------------------------------
    """

    def get_n_actions(self, agent_cls_name):
        """This component is passive: it does not add any actions."""
        return

    def get_additional_state_fields(self, agent_cls_name):
        """This component does not add any state fields."""
        return {}

    def component_step(self):
        """
        See base_component.py for detailed description.

        """
        world = self.world

        agent_rewards = np.array([agent.total_endowment("Coin") for agent in self.world.agents])

        action = self._trainPlanner.get_action(agent_rewards, self.eps)

        if action == 0:
            self.world.teams = np.array([np.array([0,1]), np.array([2,3])])
        elif action == 1:
            self.world.teams = np.array([np.array([0,2]), np.array([1,3])])
        elif action == 2:
            self.world.teams = np.array([np.array([0,3]), np.array([1,2])])

        new_state = deepcopy(self.world.teams)
        new_state = np.argsort(new_state.flatten())
        new_state = agent_rewards[new_state]
        r = rewards.inv_income_weighted_coin_endowments(coin_endowments=agent_rewards)

        done = 0
        agent_exp = {'s': agent_rewards, 'a': action, 'r': r, 's2': new_state, 'done': done}

        self._trainPlanner.add_experience(agent_exp)

        loss = self._trainPlanner.train(self._targetPlanner)
        if isinstance(loss, int):
            self._losses.append(loss)
        else:
            agent._losses.append(loss.numpy())
        self._iter+=1
        if self._iter % self._copy_step == 0:
            self._targetPlanner.copy_weights(self._trainPlanner)


    def generate_observations(self):
        """This component does not add any observations."""
        obs = {}
        return obs

    def generate_masks(self, completions=0):
        """Passive component. Masks are empty."""
        masks = {}
        return masks







# redistribute inventory within a team
@component_registry.add
class ResourceRedistribution(BaseComponent):
    """Redistributes the total coin of the mobile agents as evenly as possible.

    Note:
        If this component is used, it should always be the last component in the order!
    """

    name = "ResourceRedistribution"
    required_entities = ["Wood", "Stone"]
    agent_subclasses = ["BasicMobileAgent"]


    def __init__(
        self,
        *base_component_args,
        period=1,
        teams=None,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        # self.teams = teams
        # self.team_membership = {0:self.teams[0], #team: indices
        #                     1:self.teams[1]}
        #
        # self.team_membership_agent = {0:0, #indices: team
        #                         1:0,
        #                         2:1,
        #                         3:1}

        self.world.teams = teams

    """
    Required methods for implementing components
    --------------------------------------------
    """

    def get_n_actions(self, agent_cls_name):
        """This component is passive: it does not add any actions."""
        return

    def get_additional_state_fields(self, agent_cls_name):
        """This component does not add any state fields."""
        return {}

    def component_step(self):
        """
        See base_component.py for detailed description.

        Redistributes inventory coins so that all agents have equal coin endowment.
        """
        world = self.world

        inventory = {0:{"Wood":0,"Stone":0},
                        1:{"Wood":0,"Stone":0}}

        for agent in world.agents:
            idx = np.where(self.world.teams.flatten() == agent.idx)[0][0]
            if idx < 2: team_num = 0
            else: team_num = 1

            # inventory[self.team_membership_agent[agent.idx]]["Wood"]+=agent.state["inventory"]["Wood"]
            # inventory[self.team_membership_agent[agent.idx]]["Stone"]+=agent.state["inventory"]["Stone"]
            inventory[team_num]["Wood"]+=agent.state["inventory"]["Wood"]
            inventory[team_num]["Stone"]+=agent.state["inventory"]["Stone"]

        # print(inventory)

        for agent in world.agents:
            idx = np.where(self.world.teams.flatten() == agent.idx)[0][0]
            if idx < 2: team_num = 0
            else: team_num = 1

            # agent.state["inventory"]["Wood"] = inventory[self.team_membership_agent[agent.idx]]["Wood"] // 2
            # agent.state["inventory"]["Stone"] = inventory[self.team_membership_agent[agent.idx]]["Stone"] // 2

            if inventory[team_num]["Wood"]%2 == 0:
                agent.state["inventory"]["Wood"] = inventory[team_num]["Wood"] // 2
            if inventory[team_num]["Stone"]%2 == 0:
                agent.state["inventory"]["Stone"] = inventory[team_num]["Stone"] // 2

        # for agent in world.agents:
        #     print(agent.idx, " wood: ",agent.state["inventory"]["Wood"])
        #     print(agent.idx, " stone: ",agent.state["inventory"]["Stone"])
        #
        # exit()


    def generate_observations(self):
        """This component does not add any observations."""
        obs = {}
        return obs

    def generate_masks(self, completions=0):
        """Passive component. Masks are empty."""
        masks = {}
        return masks



@component_registry.add
class TeamTaxBracket(BaseComponent):
    """Periodically collect income taxes from agents and do lump-sum redistribution.

    Note:
        If this component is used, it should always be the last component in the order!

    Args:
        disable_taxes (bool): Whether to disable any tax collection, effectively
            enforcing that tax rates are always 0. Useful for removing taxes without
            changing the observation space. Default is False (taxes enabled).
        tax_model (str): Which tax model to use for setting taxes.
            "model_wrapper" (default) uses the actions of the planner agent;
            "us-federal-single-filer-2018-scaled" uses US federal tax rates from 2018;
            "fixed-bracket-rates" uses the rates supplied in fixed_bracket_rates.
        period (int): Length of a tax period in environment timesteps. Taxes are
            updated at the start of each period and collected/redistributed at the
            end of each period. Must be > 0. Default is 100 timesteps.
        rate_min (float): Minimum tax rate within a bracket. Must be >= 0 (default).
        rate_max (float): Maximum tax rate within a bracket. Must be <= 1 (default).
        rate_disc (float): (Only applies for "model_wrapper") the interval separating
            discrete tax rates that the planner can select. Default of 0.05 means,
            for example, the planner can select among [0.0, 0.05, 0.10, ... 1.0].
            Must be > 0 and < 1.
        n_brackets (int): How many tax brackets to use. Must be >=2. Default is 5.
        top_bracket_cutoff (float): The income at the left end of the last tax
            bracket. Must be >= 10. Default is 100 coin.
        usd_scaling (float): Scale by which to divide the US Federal bracket cutoffs
            when using bracket_spacing = "us-federal". Must be > 0. Default is 1000.
        bracket_spacing (str): How bracket cutoffs should be spaced.
            "us-federal" (default) uses scaled cutoffs from the 2018 US Federal
                taxes, with scaling set by usd_scaling (ignores n_brackets and
                top_bracket_cutoff);
            "linear" linearly spaces the n_bracket cutoffs between 0 and
                top_bracket_cutoff;
            "log" is similar to "linear" but with logarithmic spacing.
        fixed_bracket_rates (list): Required if tax_model=="fixed-bracket-rates". A
            list of fixed marginal rates to use for each bracket. Length must be
            equal to the number of brackets (7 for "us-federal" spacing, n_brackets
            otherwise).
        tax_annealing_schedule (list, optional): A length-2 list of
            [tax_annealing_warmup, tax_annealing_slope] describing the tax annealing
            schedule. See annealed_tax_mask function for details. Default behavior is
            no tax annealing.
    """

    name = "TeamTaxBracket"
    component_type = "PeriodicTax"
    required_entities = ["Coin"]
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]

    # print(">>> Component added: ", name)

    def __init__(
        self,
        *base_component_args,
        disable_taxes=False,
        tax_model="model_wrapper",
        period=100,
        rate_min=0.0,
        rate_max=1.0,
        rate_disc=0.05,
        n_brackets=5,
        top_bracket_cutoff=100,
        usd_scaling=1000.0,
        bracket_spacing="us-federal",
        fixed_bracket_rates=None,
        tax_annealing_schedule=None,
        teams=None,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        # Whether to turn off taxes. Disabling taxes will prevent any taxes from
        # being collected but the observation space will be the same as if taxes were
        # enabled, which can be useful for controlled tax/no-tax comparisons.
        self.disable_taxes = bool(disable_taxes)

        # How to set taxes
        self.tax_model = tax_model
        assert self.tax_model in [
            "model_wrapper",
            "us-federal-single-filer-2018-scaled",
            "fixed-bracket-rates",
        ]

        # How many timesteps a tax period lasts
        self.period = int(period)
        assert self.period > 0

        # Minimum marginal bracket rate
        self.rate_min = 0.0 if self.disable_taxes else float(rate_min)
        # Maximum marginal bracket rate
        self.rate_max = 0.0 if self.disable_taxes else float(rate_max)
        assert 0 <= self.rate_min <= self.rate_max <= 1.0

        # Interval for discretizing tax rate options
        # (only applies if tax_model == "model_wrapper")
        self.rate_disc = float(rate_disc)

        self.use_discretized_rates = self.tax_model == "model_wrapper"

        if self.use_discretized_rates:
            self.disc_rates = np.arange(
                self.rate_min, self.rate_max + self.rate_disc, self.rate_disc
            )
            self.disc_rates = self.disc_rates[self.disc_rates <= self.rate_max]
            assert len(self.disc_rates) > 1 or self.disable_taxes
            self.n_disc_rates = len(self.disc_rates)
        else:
            self.disc_rates = None
            self.n_disc_rates = 0

        # === income bracket definitions ===
        self.n_brackets = int(n_brackets)
        assert self.n_brackets >= 2

        self.top_bracket_cutoff = float(top_bracket_cutoff)
        assert self.top_bracket_cutoff >= 10

        self.usd_scale = float(usd_scaling)
        assert self.usd_scale > 0

        self.bracket_spacing = bracket_spacing.lower()
        assert self.bracket_spacing in ["linear", "log", "us-federal"]

        if self.bracket_spacing == "linear":
            self.bracket_cutoffs = np.linspace(
                0, self.top_bracket_cutoff, self.n_brackets
            )

        elif self.bracket_spacing == "log":
            b0_max = self.top_bracket_cutoff / (2 ** (self.n_brackets - 2))
            self.bracket_cutoffs = np.concatenate(
                [
                    [0],
                    2
                    ** np.linspace(
                        np.log2(b0_max),
                        np.log2(self.top_bracket_cutoff),
                        n_brackets - 1,
                    ),
                ]
            )
        elif self.bracket_spacing == "us-federal":
            self.bracket_cutoffs = (
                np.array([0, 9700, 39475, 84200, 160725, 204100, 510300])
                / self.usd_scale
            )
            self.n_brackets = len(self.bracket_cutoffs)
            self.top_bracket_cutoff = float(self.bracket_cutoffs[-1])
        else:
            raise NotImplementedError

        self.bracket_edges = np.concatenate([self.bracket_cutoffs, [np.inf]])
        self.bracket_sizes = self.bracket_edges[1:] - self.bracket_edges[:-1]

        assert self.bracket_cutoffs[0] == 0

        if self.tax_model == "us-federal-single-filer-2018-scaled":
            assert self.bracket_spacing == "us-federal"

        if self.tax_model == "fixed-bracket-rates":
            assert isinstance(fixed_bracket_rates, (tuple, list))
            assert np.min(fixed_bracket_rates) >= 0
            assert np.max(fixed_bracket_rates) <= 1
            assert len(fixed_bracket_rates) == self.n_brackets
            self._fixed_bracket_rates = np.array(fixed_bracket_rates)
        else:
            self._fixed_bracket_rates = None

        # === bracket tax rates ===
        self.curr_bracket_tax_rates = np.zeros_like(self.bracket_cutoffs)
        self.curr_rate_indices = [0 for _ in range(self.n_brackets)]

        # === tax cycle definitions ===
        self.tax_cycle_pos = 1
        self.last_coin = [0 for _ in range(self.n_agents)]
        self.last_income = [0 for _ in range(self.n_agents)]
        self.last_marginal_rate = [0 for _ in range(self.n_agents)]
        self.last_effective_tax_rate = [0 for _ in range(self.n_agents)]

        # === trackers ===
        self.total_collected_taxes = 0
        self.all_effective_tax_rates = []
        self._schedules = {"{:03d}".format(int(r)): [0] for r in self.bracket_cutoffs}
        self._occupancy = {"{:03d}".format(int(r)): 0 for r in self.bracket_cutoffs}
        self.taxes = []

        # === tax annealing ===
        # for annealing of non-planner max taxes
        self._annealed_rate_max = float(self.rate_max)
        self._last_completions = 0

        # === for teams ===
        self._teams = teams
        self.world.teams = teams

        # for annealing of planner actions
        self.tax_annealing_schedule = tax_annealing_schedule
        if tax_annealing_schedule is not None:
            assert isinstance(self.tax_annealing_schedule, (tuple, list))
            self._annealing_warmup = self.tax_annealing_schedule[0]
            self._annealing_slope = self.tax_annealing_schedule[1]
            self._annealed_rate_max = annealed_tax_limit(
                self._last_completions,
                self._annealing_warmup,
                self._annealing_slope,
                self.rate_max,
            )
        else:
            self._annealing_warmup = None
            self._annealing_slope = None

        if self.tax_model == "model_wrapper" and not self.disable_taxes:
            planner_action_tuples = self.get_n_actions("BasicPlanner")
            self._planner_tax_val_dict = {
                k: self.disc_rates for k, v in planner_action_tuples
            }
        else:
            self._planner_tax_val_dict = {}
        self._planner_masks = None

        # === placeholders ===
        self._curr_rates_obs = np.array(self.curr_marginal_rates)
        self._last_income_obs = np.array(self.last_income) / self.period
        self._last_income_obs_sorted = self._last_income_obs[
            np.argsort(self._last_income_obs)
        ]

    # Methods for getting/setting marginal tax rates
    # ----------------------------------------------

    # ------- US Federal taxes
    @property
    def us_federal_single_filer_2018_scaled(self):
        """
        https://turbotax.intuit.com/tax-tips/irs-tax-return/current-federal-tax-rate-schedules/L7Bjs1EAD
        If taxable income is over—
        but not over—
        the tax is:
        $0
        $9,700
        10% of the amount over $0
        $9,700
        $39,475
        $970 plus 12% of the amount over $9,700
        $39,475
        $84,200
        $4,543 plus 22% of the amount over $39,475
        $84,200
        $160,725
        $14,382 plus 24% of the amount over $84,200
        $160,725
        $204,100
        $32,748 plus 32% of the amount over $160,725
        $204,100
        $510,300
        $46,628 plus 35% of the amount over $204,100
        $510,300
        no limit
        $153,798 plus 37% of the amount over $510,300
        """
        return [0.1, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]

    # ------- fixed-bracket-rates
    @property
    def fixed_bracket_rates(self):
        """Return whatever fixed bracket rates were set during initialization."""
        return self._fixed_bracket_rates

    @property
    def curr_rate_max(self):
        """Maximum allowable tax rate, given current progress of any tax annealing."""
        if self.tax_annealing_schedule is None:
            return self.rate_max
        return self._annealed_rate_max

    @property
    def curr_marginal_rates(self):
        """The current set of marginal tax bracket rates."""
        if self.use_discretized_rates:
            return self.disc_rates[self.curr_rate_indices]

        if self.tax_model == "us-federal-single-filer-2018-scaled":
            return np.minimum(
                np.array(self.us_federal_single_filer_2018_scaled), self.curr_rate_max
            )

        if self.tax_model == "fixed-bracket-rates":
            return np.minimum(np.array(self.fixed_bracket_rates), self.curr_rate_max)

        raise NotImplementedError

    def set_new_period_rates_model(self):
        """Update taxes using actions from the tax model."""
        if self.disable_taxes:
            return

        # AI version
        for i, bracket in enumerate(self.bracket_cutoffs):
            planner_action = self.world.planner.get_component_action(
                self.name, "TaxIndexBracket_{:03d}".format(int(bracket))
            )
            if planner_action == 0:
                pass
            elif planner_action <= self.n_disc_rates:
                self.curr_rate_indices[i] = int(planner_action - 1)
            else:
                raise ValueError

    # Methods for collecting and redistributing taxes
    # -----------------------------------------------

    def income_bin(self, income):
        """Return index of tax bin in which income falls."""
        if income < 0:
            return 0.0
        meets_min = income >= self.bracket_edges[:-1]
        under_max = income < self.bracket_edges[1:]
        bracket_bool = meets_min * under_max
        return self.bracket_cutoffs[np.argmax(bracket_bool)]

    def marginal_rate(self, income):
        """Return the marginal tax rate applied at this income level."""
        if income < 0:
            return 0.0
        meets_min = income >= self.bracket_edges[:-1]
        under_max = income < self.bracket_edges[1:]
        bracket_bool = meets_min * under_max
        return self.curr_marginal_rates[np.argmax(bracket_bool)]

    def taxes_due(self, income):
        """Return the total amount of taxes due at this income level."""
        past_cutoff = np.maximum(0, income - self.bracket_cutoffs)
        bin_income = np.minimum(self.bracket_sizes, past_cutoff)
        bin_taxes = self.curr_marginal_rates * bin_income
        return np.sum(bin_taxes)

    # def enact_taxes(self):
    #     """Calculate period income & tax burden. Collect taxes and redistribute."""
    #     net_tax_revenue = 0
    #     tax_dict = dict(
    #         schedule=np.array(self.curr_marginal_rates),
    #         cutoffs=np.array(self.bracket_cutoffs),
    #     )
    #
    #     for curr_rate, bracket_cutoff in zip(
    #         self.curr_marginal_rates, self.bracket_cutoffs
    #     ):
    #         self._schedules["{:03d}".format(int(bracket_cutoff))].append(
    #             float(curr_rate)
    #         )
    #
    #     self.last_income = []
    #     self.last_effective_tax_rate = []
    #     self.last_marginal_rate = []
    #     for agent, last_coin in zip(self.world.agents, self.last_coin):
    #         income = agent.total_endowment("Coin") - last_coin
    #         tax_due = self.taxes_due(income)
    #         effective_taxes = np.minimum(
    #             agent.state["inventory"]["Coin"], tax_due
    #         )  # Don't take from escrow.
    #         marginal_rate = self.marginal_rate(income)
    #         effective_tax_rate = float(effective_taxes / np.maximum(0.000001, income))
    #         tax_dict[str(agent.idx)] = dict(
    #             income=float(income),
    #             tax_paid=float(effective_taxes),
    #             marginal_rate=marginal_rate,
    #             effective_rate=effective_tax_rate,
    #         )
    #
    #         # Actually collect the taxes
    #         agent.state["inventory"]["Coin"] -= effective_taxes
    #         net_tax_revenue += effective_taxes
    #
    #         self.last_income.append(float(income))
    #         self.last_marginal_rate.append(float(marginal_rate))
    #         self.last_effective_tax_rate.append(effective_tax_rate)
    #
    #         self.all_effective_tax_rates.append(effective_tax_rate)
    #         self._occupancy["{:03d}".format(int(self.income_bin(income)))] += 1
    #
    #     self.total_collected_taxes += float(net_tax_revenue)
    #
    #     lump_sum = net_tax_revenue / self.n_agents
    #     for agent in self.world.agents:
    #         agent.state["inventory"]["Coin"] += lump_sum
    #         tax_dict[str(agent.idx)]["lump_sum"] = float(lump_sum)
    #         self.last_coin[agent.idx] = float(agent.total_endowment("Coin"))
    #
    #     self.taxes.append(tax_dict)
    #
    #     # Pre-compute some things that will be useful for generating observations
    #     self._last_income_obs = np.array(self.last_income) / self.period
    #     self._last_income_obs_sorted = self._last_income_obs[
    #         np.argsort(self._last_income_obs)
    #     ]

    def enact_team_taxes(self):
        """Calculate period income & tax burden. Collect taxes and redistribute within each agent team."""
        net_tax_revenue = 0
        tax_dict = dict(
            schedule=np.array(self.curr_marginal_rates),
            cutoffs=np.array(self.bracket_cutoffs),
        )

        for curr_rate, bracket_cutoff in zip(
            self.curr_marginal_rates, self.bracket_cutoffs
        ):
            self._schedules["{:03d}".format(int(bracket_cutoff))].append(
                float(curr_rate)
            )


        # self.last_income = []
        # self.last_effective_tax_rate = []
        # self.last_marginal_rate = []
        self.last_income = [0 for _ in range(self.n_agents)]
        self.last_marginal_rate = [0 for _ in range(self.n_agents)]
        self.last_effective_tax_rate = [0 for _ in range(self.n_agents)]

        self.team_tax = np.zeros((self._teams.shape[0],))

        # for team_num, team in enumerate(self._teams):
        for team_num, team in enumerate(self.world.teams):
            for agent_idx in team:
        # for agent, last_coin in zip(self.world.agents, self.last_coin):
                agent = self.world.agents[agent_idx]
                last_coin = self.last_coin[agent_idx]

                income = agent.total_endowment("Coin") - last_coin
                tax_due = self.taxes_due(income)
                effective_taxes = np.minimum(
                    agent.state["inventory"]["Coin"], tax_due
                )  # Don't take from escrow.
                marginal_rate = self.marginal_rate(income)
                effective_tax_rate = float(effective_taxes / np.maximum(0.000001, income))
                tax_dict[str(agent.idx)] = dict(
                    income=float(income),
                    tax_paid=float(effective_taxes),
                    marginal_rate=marginal_rate,
                    effective_rate=effective_tax_rate,
                )

                # Actually collect the taxes
                agent.state["inventory"]["Coin"] -= effective_taxes
                # net_tax_revenue += effective_taxes
                self.team_tax[team_num] += effective_taxes

                self.last_income[agent_idx] = float(income)
                self.last_marginal_rate[agent_idx] = float(marginal_rate)
                self.last_effective_tax_rate[agent_idx] = effective_tax_rate

                self.all_effective_tax_rates.append(effective_tax_rate)
                self._occupancy["{:03d}".format(int(self.income_bin(income)))] += 1

# updated to here
        self.total_collected_taxes += float(net_tax_revenue)

        lump_sum = np.zeros((self._teams.shape[0],))
        for team_num, team in enumerate(self._teams):
            lump_sum[team_num] = self.team_tax[team_num] / self._teams.shape[1]

        # for agent in self.world.agents:
        # for team_num, team in enumerate(self._teams):
        for team_num, team in enumerate(self.world.teams):
            for agent_idx in team:
                agent = self.world.agents[agent_idx]

                agent.state["inventory"]["Coin"] += lump_sum[team_num]
                tax_dict[str(agent.idx)]["lump_sum"] = float(lump_sum[team_num])
                self.last_coin[agent.idx] = float(agent.total_endowment("Coin"))

        self.taxes.append(tax_dict)

        # Pre-compute some things that will be useful for generating observations
        self._last_income_obs = np.array(self.last_income) / self.period
        self._last_income_obs_sorted = self._last_income_obs[
            np.argsort(self._last_income_obs)
        ]

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        If using the "model_wrapper" tax model and taxes are enabled, the planner's
        action space includes an action subspace for each of the tax brackets. Each
        such action space has as many actions as there are discretized tax rates.
        """
        # Only the planner takes actions through this component
        if agent_cls_name == "BasicPlanner":
            if self.tax_model == "model_wrapper" and not self.disable_taxes:
                # For every bracket, the planner can select one of the discretized
                # tax rates.
                return [
                    ("TaxIndexBracket_{:03d}".format(int(r)), self.n_disc_rates)
                    for r in self.bracket_cutoffs
                ]

        # Return 0 (no added actions) if the other conditions aren't met
        return 0

    def get_additional_state_fields(self, agent_cls_name):
        """This component does not add any agent state fields."""
        return {}

    def component_step(self):
        """
        See base_component.py for detailed description.

        On the first day of each tax period, update taxes. On the last day, enact them.
        """

        # 1. On the first day of a new tax period: Set up the taxes for this period.
        if self.tax_cycle_pos == 1:
            if self.tax_model == "model_wrapper":
                self.set_new_period_rates_model()

            self._curr_rates_obs = np.array(self.curr_marginal_rates)

        # 2. On the last day of the tax period: Get $-taxes AND update agent endowments
        if self.tax_cycle_pos >= self.period:
            # if self._teams is not None:
            self.enact_team_taxes()
            # else:
            #     self.enact_taxes()      # this is where redistribution happens
            self.tax_cycle_pos = 0

        else:
            self.taxes.append([])

        # increment timestep
        self.tax_cycle_pos += 1

    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Agents observe where in the tax period cycle they are, information about the
        last period's incomes, and the current marginal tax rates, including the
        marginal rate that will apply to their next unit of income.

        The planner observes the same type of information, but for all of the agents. It
        also sees, for each agent, their marginal tax rate and reported income from
        the previous tax period.
        """
        is_tax_day = float(self.tax_cycle_pos >= self.period)
        is_first_day = float(self.tax_cycle_pos == 1)
        tax_phase = self.tax_cycle_pos / self.period

        obs = dict()

        obs[self.world.planner.idx] = dict(
            is_tax_day=is_tax_day,
            is_first_day=is_first_day,
            tax_phase=tax_phase,
            last_incomes=self._last_income_obs_sorted,
            curr_rates=self._curr_rates_obs,
        )

        for agent in self.world.agents:
            i = agent.idx
            k = str(i)

            curr_marginal_rate = self.marginal_rate(
                agent.total_endowment("Coin") - self.last_coin[i]
            )

            obs[k] = dict(
                is_tax_day=is_tax_day,
                is_first_day=is_first_day,
                tax_phase=tax_phase,
                last_incomes=self._last_income_obs_sorted,
                curr_rates=self._curr_rates_obs,
                marginal_rate=curr_marginal_rate,
            )

            obs["p" + k] = dict(
                last_income=self._last_income_obs[i],
                last_marginal_rate=self.last_marginal_rate[i],
                curr_marginal_rate=curr_marginal_rate,
            )

        return obs

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.

        Masks only apply to the planner and if tax_model == "model_wrapper" and taxes
        are enabled.
        All tax actions are masked (so, only NO-OPs can be sampled) on all timesteps
        except when self.tax_cycle_pos==1 (meaning a new tax period is starting).
        When self.tax_cycle_pos==1, tax actions are masked in order to enforce any
        tax annealing.
        """
        if (
            completions != self._last_completions
            and self.tax_annealing_schedule is not None
        ):
            self._last_completions = int(completions)
            self._annealed_rate_max = annealed_tax_limit(
                completions,
                self._annealing_warmup,
                self._annealing_slope,
                self.rate_max,
            )

        if self.disable_taxes:
            return {}

        if self.tax_model == "model_wrapper":
            # No annealing. Generate masks using default method.
            if self.tax_annealing_schedule is None:
                if self._planner_masks is None:
                    masks = super().generate_masks(completions=completions)
                    self._planner_masks = dict(
                        new_taxes=deepcopy(masks[self.world.planner.idx]),
                        zeros={
                            k: np.zeros_like(v)
                            for k, v in masks[self.world.planner.idx].items()
                        },
                    )

                # No need to recompute. Use the cached masks.
                masks = dict()
                if self.tax_cycle_pos != 1 or self.disable_taxes:
                    # Apply zero masks for any timestep where taxes
                    # are not going to be updated.
                    masks[self.world.planner.idx] = self._planner_masks["zeros"]
                else:
                    masks[self.world.planner.idx] = self._planner_masks["new_taxes"]

            # Doing annealing.
            else:
                # Figure out what the masks should be this episode.
                if self._planner_masks is None:
                    planner_masks = {
                        k: annealed_tax_mask(
                            completions,
                            self._annealing_warmup,
                            self._annealing_slope,
                            tax_values,
                        )
                        for k, tax_values in self._planner_tax_val_dict.items()
                    }
                    self._planner_masks = dict(
                        new_taxes=deepcopy(planner_masks),
                        zeros={k: np.zeros_like(v) for k, v in planner_masks.items()},
                    )

                # No need to recompute. Use the cached masks.
                masks = dict()
                if self.tax_cycle_pos != 1 or self.disable_taxes:
                    # Apply zero masks for any timestep where taxes
                    # are not going to be updated.
                    masks[self.world.planner.idx] = self._planner_masks["zeros"]
                else:
                    masks[self.world.planner.idx] = self._planner_masks["new_taxes"]

        # We are not using a learned planner. Generate masks by the default method.
        else:
            masks = super().generate_masks(completions=completions)

        return masks

    # For non-required customization
    # ------------------------------

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Reset trackers.
        """
        self.curr_rate_indices = [0 for _ in range(self.n_brackets)]

        self.tax_cycle_pos = 1
        self.last_coin = [
            float(agent.total_endowment("Coin")) for agent in self.world.agents
        ]
        self.last_income = [0 for _ in range(self.n_agents)]
        self.last_marginal_rate = [0 for _ in range(self.n_agents)]
        self.last_effective_tax_rate = [0 for _ in range(self.n_agents)]

        self._curr_rates_obs = np.array(self.curr_marginal_rates)
        self._last_income_obs = np.array(self.last_income) / self.period
        self._last_income_obs_sorted = self._last_income_obs[
            np.argsort(self._last_income_obs)
        ]

        self.taxes = []
        self.total_collected_taxes = 0
        self.all_effective_tax_rates = []
        self._schedules = {"{:03d}".format(int(r)): [] for r in self.bracket_cutoffs}
        self._occupancy = {"{:03d}".format(int(r)): 0 for r in self.bracket_cutoffs}
        self._planner_masks = None

    def get_metrics(self):
        """
        See base_component.py for detailed description.

        Return metrics related to bracket rates, bracket occupancy, and tax collection.
        """
        out = dict()

        n_observed_incomes = np.maximum(1, np.sum(list(self._occupancy.values())))
        for c in self.bracket_cutoffs:
            k = "{:03d}".format(int(c))
            out["avg_bracket_rate/{}".format(k)] = np.mean(self._schedules[k])
            out["bracket_occupancy/{}".format(k)] = (
                self._occupancy[k] / n_observed_incomes
            )

        if not self.disable_taxes:
            out["avg_effective_tax_rate"] = np.mean(self.all_effective_tax_rates)
            out["total_collected_taxes"] = float(self.total_collected_taxes)

            # Indices of richest and poorest agents
            agent_coin_endows = np.array(
                [agent.total_endowment("Coin") for agent in self.world.agents]
            )
            idx_poor = np.argmin(agent_coin_endows)
            idx_rich = np.argmax(agent_coin_endows)

            tax_days = self.taxes[(self.period - 1) :: self.period]
            for i, tag in zip([idx_poor, idx_rich], ["poorest", "richest"]):
                total_income = np.maximum(
                    0, [tax_day[str(i)]["income"] for tax_day in tax_days]
                ).sum()
                total_tax_paid = np.sum(
                    [tax_day[str(i)]["tax_paid"] for tax_day in tax_days]
                )
                # Report the overall tax rate over the episode
                # for the richest and poorest agents
                out["avg_tax_rate/{}".format(tag)] = total_tax_paid / np.maximum(
                    0.001, total_income
                )

        return out

    def get_dense_log(self):
        """
        Log taxes.

        Returns:
            taxes (list): A list of tax collections. Each entry corresponds to a single
                timestep. Entries are empty except for timesteps where a tax period
                ended and taxes were collected. For those timesteps, each entry
                contains the tax schedule, each agent's reported income, tax paid,
                and redistribution received.
                Returns None if taxes are disabled.
        """
        if self.disable_taxes:
            return None
        return self.taxes
