from manim import *
import os, random
import itertools as it

# for some reason PATH didnt include the miktek location, so manually add it
# os.environ["PATH"] = os.environ["PATH"] + r";C:\Users\user\AppData\Local\Programs\MiKTeX\miktex\bin\x64\\"


class GrowArrowCustom(GrowArrow):
    """
    Custom Anim that grows arrow from current length to end length
    """
    def __init__(self, arrow: Arrow, end:float, **kwargs):
        self.start_len = arrow.get_length()

        super().__init__(arrow, **kwargs)

        arrow.scale(end / self.start_len, scale_tips=False, about_point=self.point)

    def create_starting_mobject(self) -> Mobject:
        start_arrow = self.mobject.copy()
        length = start_arrow.get_length()
        start_arrow.scale(self.start_len/length, scale_tips=False, about_point=self.point)
        if self.point_color:
            start_arrow.set_color(self.point_color)
        return start_arrow


# A customizable Sequential Neural Network
class NeuralNetworkMobject(VGroup):
    CONFIG = {
        "neuron_radius": 0.15,
        "neuron_to_neuron_buff": MED_SMALL_BUFF,
        "layer_to_layer_buff": LARGE_BUFF,
        "output_neuron_color": ORANGE,
        "input_neuron_color": ORANGE,
        "hidden_layer_neuron_color": WHITE,
        "neuron_stroke_width": 2,
        "neuron_fill_color": GREEN,
        "edge_color": WHITE,
        "edge_stroke_width": 2,
        "edge_propogation_color": YELLOW,
        "edge_propogation_time": 1,
        "max_shown_neurons": 16,
        "brace_for_large_layers": True,
        "average_shown_activation_of_large_layer": True,
        "include_output_labels": False,
        "arrow": False,
        "arrow_tip_size": 0.1,
        "left_size": 1,
        "neuron_fill_opacity": 1
    }

    # Constructor with parameters of the neurons in a list
    def __init__(self, neural_network, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.__dict__.update(self.CONFIG)
        self.layer_sizes = neural_network
        self.add_neurons()
        self.add_edges()
        self.add_to_back(self.layers)

    # Helper method for constructor
    def add_neurons(self):
        layers = VGroup(*[
            self.get_layer(size, index)
            for index, size in enumerate(self.layer_sizes)
        ])
        layers.arrange_submobjects(RIGHT, buff=self.layer_to_layer_buff)
        self.layers = layers


    # Helper method for constructor
    def get_nn_fill_color(self, index):
        if index == -1 or index == len(self.layer_sizes) - 1:
            return self.output_neuron_color
        if index == 0:
            return self.input_neuron_color
        else:
            return self.hidden_layer_neuron_color

    # Helper method for constructor
    def get_layer(self, size, index=-1):
        layer = VGroup()
        n_neurons = size
        if n_neurons > self.max_shown_neurons:
            n_neurons = self.max_shown_neurons
        neurons = VGroup(*[
            Circle(
                radius=self.neuron_radius,
                stroke_color=self.get_nn_fill_color(index),
                stroke_width=self.neuron_stroke_width,
                fill_color=BLACK,
                fill_opacity=self.neuron_fill_opacity,
            )
            for x in range(n_neurons)
        ])
        neurons.arrange_submobjects(
            DOWN, buff=self.neuron_to_neuron_buff
        )
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        if size > n_neurons:
            dots = Tex("\\vdots")
            dots.move_to(neurons)
            VGroup(*neurons[:len(neurons) // 2]).next_to(
                dots, UP, MED_SMALL_BUFF
            )
            VGroup(*neurons[len(neurons) // 2:]).next_to(
                dots, DOWN, MED_SMALL_BUFF
            )
            layer.dots = dots
            layer.add(dots)
            if self.brace_for_large_layers:
                brace = Brace(layer, LEFT)
                brace_label = brace.get_tex(str(size))
                layer.brace = brace
                layer.brace_label = brace_label
                layer.add(brace, brace_label)

        return layer

    # Helper method for constructor
    def add_edges(self):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                edge = self.get_edge(n1, n2)
                edge_group.add(edge)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
            self.edge_groups.add(edge_group)
        self.add_to_back(self.edge_groups)

    # Helper method for constructor
    def get_edge(self, neuron1, neuron2):
        if self.arrow:
            return Arrow(
                neuron1.get_center(),
                neuron2.get_center(),
                buff=self.neuron_radius,
                stroke_color=self.edge_color,
                stroke_width=self.edge_stroke_width,
                tip_length=self.arrow_tip_size
            )
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff=self.neuron_radius,
            stroke_color=self.edge_color,
            stroke_width=self.edge_stroke_width,
        )

    # Labels each input neuron with a char l
    def label_inputs(self, l):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[0].neurons):
            label = Text(l + str(n))
            label.set_height(0.3 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    # Labels each output neuron with a char l
    def label_outputs(self, l):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = Text(l + str(n))
            label.set_height(0.3 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    # Labels the hidden layers with a char l
    def label_hidden_layers(self, l):
        self.output_labels = VGroup()
        for layer in self.layers[1:-1]:
            for n, neuron in enumerate(layer.neurons):
                label = Text(l)
                label.set_height(0.4 * neuron.get_height())
                label.move_to(neuron)
                self.output_labels.add(label)
        self.add(self.output_labels)


class QLearningExplained(Scene):
    def construct(self):
        text = Text("Making a self driving car using Deep Q Learning", font_size=42, color=WHITE)
        text2 = Text("Part 1", font_size=36, color=ORANGE)
        text.set_y(0.25)
        text2.set_y(-0.3)
        self.play(Write(text), run_time=3)
        self.play(Write(text2), run_time=2)
        self.wait(1)
        self.play(Unwrite(text), Unwrite(text2), run_time=1)

        # insert clip of self driving car
        self.wait(1)

        title = Text("What is Reinforcement Learning?")
        body = Tex(
            r"Reinforcement Learning is a method to train an ",
            r"agent\\",
            r" (the car) to create an ",
            r"optimal policy",
            r"."
        )

        body.set_color_by_tex("agent", ORANGE)
        body.set_color_by_tex("optimal policy", ORANGE)

        title.move_to(Point(location=(-2, 3, 0)))

        self.play(Write(title))
        self.wait(1)
        self.play(Write(body, run_time=3))

        self.wait(4.5)

        self.play(body.animate.move_to(Point(location=(0, -8, 0))))
        self.remove(body)

        definitions_size = 36
        env_text = Text("Environment: the world that the agent will navigate", t2c={"Environment": ORANGE}, font_size=definitions_size)
        env_text.set_y(1)
        state_text = Text("State: the observations that the agent has of the environment", t2c={"State": ORANGE}, font_size=definitions_size)
        state_text.set_y(0)
        actions_text = Text("Actions: the choices that the agent can take to change its state", t2c={"Actions": ORANGE}, font_size=definitions_size)
        actions_text.set_y(-1)
        policy_text = Text("Policy: how the agent chooses its actions", t2c={"Policy": ORANGE}, font_size=definitions_size)
        policy_text.set_y(-2)

        self.play(
            AnimationGroup(
                Write(env_text, run_time=2),
                Write(state_text, run_time=3),
                Write(actions_text, run_time=2.5),
                Write(policy_text),
                lag_ratio=1.6
            )
        )

        self.wait(5)

        self.play(
            AnimationGroup(
                Unwrite(env_text),
                Unwrite(state_text),
                Unwrite(actions_text),
                Unwrite(policy_text),
                lag_ratio=0.2, run_time=0.8
            )
        )

        center = title.get_left()

        title2 = Text("How is this done?")
        title2.set_x(center[0]+title2.width/2)
        title2.set_y(center[1])

        how_text = Text(
            "If the agent makes a good action,\n\t it is rewarded.\n\nIf the agent makes a bad action,\n\t it is punished (negative reward).",
            t2c={"good": GREEN, "bad": RED}, font_size=44
        )
        how_text.set_y(-0.2)

        s = SurroundingRectangle(
            how_text,
            ORANGE,
            corner_radius=0.2
        )

        s.scale(1.2)

        self.play(ReplacementTransform(title, title2))
        self.wait(1)
        self.play(Write(how_text), Write(s))
        self.wait(4)
        self.play(Unwrite(how_text), Unwrite(s), run_time=0.5)
        self.wait(0.5)

        center = title2.get_left()
        title3 = Text("What is Deep Q Learning?")
        title3.set_x(center[0] + title3.width / 2)
        title3.set_y(center[1])

        dqn_text1 = Text("To explain Deep Q Learning, first Q Learning", t2c={"Q Learning": ORANGE, "Deep": ORANGE}, font_size=40)
        dqn_text1.set_y(0.3)
        dqn_text2 = Text("and Deep Learning should be explained.", t2c={"Deep Learning": ORANGE}, font_size=40)
        dqn_text2.set_y(-0.3)

        self.play(ReplacementTransform(title2, title3))
        self.wait(1)
        self.play(Write(dqn_text1), Write(dqn_text2))
        self.wait(5.5)

        title3_copy = title3.copy()
        self.add(title3_copy)

        title4 = Text("What is Q Learning?")

        center = title3.get_left()
        title4.set_x(center[0] + title4.width / 2)
        title4.set_y(center[1])

        title5 = Text("What is Deep Learning?")
        title5.set_x(8-title5.width/2 - 1)
        title5.set_y(center[1])

        self.play(Unwrite(dqn_text1), Unwrite(dqn_text2))
        self.play(ReplacementTransform(title3, title4), ReplacementTransform(title3_copy, title5))

        self.wait(1)

        self.play(title5.animate.set_opacity(0.5))

        q_values = Tex(r"Q Learning introduces ", r"Q values", r". For each state,\\there is a Q value for every action.",
                       font_size=46)
        q_values.set_color_by_tex("Q values", ORANGE)
        q_values.set_y(.4)

        perfect_text = Tex(r"In a perfect solution, the greatest Q value represents\\the best action for the state.",
                           font_size=46)
        perfect_text.set_y(-1.1)

        self.play(
            AnimationGroup(
                Write(q_values),
                Write(perfect_text),
                lag_ratio=2.5
            )
        )

        self.wait(7.5)

        self.play(
            AnimationGroup(
                Unwrite(q_values),
                Unwrite(perfect_text),
                lag_ratio=0.8
            )
        )

        q_table_text = Tex(r"These Q values are saved in a large table,\\mapping states to action Q values.", font_size=44)
        q_table_text.set_y(1.75)

        q_table = Table(
            [
                [""]*3,
                ["23.5", "26.0", "21.2"],
                ["35.2", "42.6", "27.9"],
                ["59.2", "53.4", "60.1"]
            ],
            row_labels=[Text("State ···", color=ORANGE), Text("State 4", color=ORANGE), Text("State 5", color=ORANGE), Text("State 6", color=ORANGE)],
            col_labels=[Text("Action 0", color=ORANGE), Text("Action 1", color=ORANGE), Text("Action 2", color=ORANGE)],
            include_outer_lines=True
        )

        q_table.scale(0.7)
        q_table.set_y(-1.5)

        self.play(
            Write(q_table),
            Write(q_table_text)
        )
        self.wait(7)
        self.play(
            Unwrite(q_table),
            Unwrite(q_table_text)
        )
        self.wait(0.5)

        self.play(
            title4.animate.set_opacity(0.5),
            title5.animate.set_opacity(1)
        )

        body = Text("Deep Learning uses a neural network to approximate a function.", font_size=29)
        body.set_y(1.7)
        body_2 = Text("A function is approximated by telling the network the inputs and the expected outputs.", font_size=24)
        body_2.set_y(1.15)

        nn = NeuralNetworkMobject(
            [3, 4, 4, 2]
        )

        nn.scale(1.5)
        nn.label_inputs("in ")
        nn.label_outputs("out ")
        nn.set_y(-1.25)

        self.play(Write(nn), Write(body), Write(body_2))

        def visualise_prop(scene, nn, forward=True):
            layers = nn.edge_groups if forward else nn.edge_groups[::-1]
            for layer in layers:
                for line in layer:
                    line.generate_target()
                    line.target.stroke_width = 3.8
                    line.target.set_color(ORANGE if forward else RED)

                a = AnimationGroup(
                    *(AnimationGroup(MoveToTarget(l, run_time=0.2)) for l in layer),
                    lag_ratio=0.2
                )

                scene.play(a)

                for line in layer:
                    line.generate_target()

                    width = 2 if forward else random.random()*4.5+0.5
                    line.target.stroke_width = width

                    col = WHITE if forward else interpolate_color(WHITE, "#ff974c", width/5)

                    line.target.set_color(col)

                b = AnimationGroup(
                    *(AnimationGroup(MoveToTarget(l, run_time=0.1)) for l in layer),
                    lag_ratio=0.2
                )

                scene.play(b)

        def label_layer(inputs,
                        values,
                        nn: NeuralNetworkMobject,
                        scene: Scene,
                        old_labels,
                        label_offset=0.7,
                        text_kwargs=None,
                        header=""):

            if text_kwargs is None:
                text_kwargs = dict()

            label_offset = -label_offset if inputs else label_offset

            labels = []
            layer = nn.layers[0] if inputs else nn.layers[-1]
            layer = layer[0]

            header_text = None
            if header:
                first_node = layer[0]
                header_text = Text(header, font_size=16, **text_kwargs)
                header_text.move_to(first_node)
                header_text.set_y(header_text.get_y() + 0.7)
                header_text.set_x(header_text.get_x() + label_offset)
                scene.play(Write(header_text))

            for mobject, value in zip(layer, values):
                label = Text(
                    str(round(value, 1)),
                    font_size=24,
                    **text_kwargs
                )
                label.move_to(mobject)
                label.set_x(label.get_x() + label_offset)
                labels.append(label)

            if len(old_labels):
                anim_group = AnimationGroup(
                    *(ReplacementTransform(old, new) for old, new in zip(old_labels, labels)),
                    lag_ratio=0.3
                )
            else:
                anim_group = AnimationGroup(
                    *(Write(l) for l in labels),
                    lag_ratio=0.3
                )
            scene.play(anim_group)
            return labels, header_text

        def get_random_values(value, length):
            return np.array([((((random.random()-0.5) / 2) * value)+value) for _ in range(length)])

        self.wait(5)

        # label initial inputs
        in_values = get_random_values(3, 3)
        inputs, input_header = label_layer(True, in_values, nn, self, [], header="Input")

        # forward prop
        visualise_prop(self, nn)

        # label outputs
        out_values = get_random_values(10, 3)
        outputs, output_header = label_layer(False, out_values, nn, self, [], header="Output")
        expected_values = get_random_values(5, 3)
        expected, expected_header = label_layer(False, expected_values, nn, self, [], 1.8, {"color": BLUE_B}, header="Expected")
        errors = out_values - expected_values
        error, error_header = label_layer(False, errors, nn, self, [], 2.8, {"color": RED}, header="Error")

        # back prop
        visualise_prop(self, nn, False)

        # label new inputs
        in_values = get_random_values(4, 3)
        inputs, _ = label_layer(True, in_values, nn, self, inputs)

        # forward prop 2
        visualise_prop(self, nn)

        # label outputs 2
        out_values = get_random_values(6, 3)
        outputs, _ = label_layer(False, out_values, nn, self, outputs)
        expected_values = get_random_values(1, 3)
        expected, _ = label_layer(False, expected_values, nn, self, expected, 1.8, {"color": BLUE_B})
        errors = out_values - expected_values
        error, _ = label_layer(False, errors, nn, self, error, 2.8, {"color": RED})

        # back prop 2
        visualise_prop(self, nn, False)

        # label new inputs 3
        in_values = get_random_values(2, 3)
        inputs, _ = label_layer(True, in_values, nn, self, inputs)

        # forward prop 3
        visualise_prop(self, nn)

        # label outputs 3
        out_values = get_random_values(5, 3)
        outputs, _ = label_layer(False, out_values, nn, self, outputs)
        expected_values = get_random_values(5, 3)
        expected, _ = label_layer(False, expected_values, nn, self, expected, 1.8, {"color": BLUE_B})
        errors = np.round(out_values, 1) - np.round(expected_values, 1)
        error, _ = label_layer(False, errors, nn, self, error, 2.8, {"color": RED})

        self.wait(3)

        center = title4.get_left()
        title6 = Text("So, what is Deep Q Learning?")
        title6.set_x(center[0] + title6.width / 2)
        title6.set_y(center[1])

        title6_1 = title6.copy()

        self.play(
            ReplacementTransform(title5, title6),
            ReplacementTransform(title4, title6_1),
        )

        self.wait(1)

        self.play(
            Unwrite(nn),
            Unwrite(body),
            Unwrite(body_2),
            AnimationGroup(
                Unwrite(input_header),
                Unwrite(output_header),
                Unwrite(expected_header),
                Unwrite(error_header),
                lag_ratio=0.3
            ),
            AnimationGroup(
                *(Unwrite(inp) for inp in inputs)
            ),
            AnimationGroup(
                *(Unwrite(output) for output in outputs)
            ),
            AnimationGroup(
                *(Unwrite(e) for e in expected)
            ),
            AnimationGroup(
                *(Unwrite(e) for e in error)
            ),
        )

        self.remove(title6_1)

        dq_explained = Tex(
            r"In Deep Q Learning, the Q values for a given state\\are approximated with a Deep Neural Network.\\This means that environments can be:",
            font_size=48,
        )
        dq_explained.set_y(1.35)

        rect_1 = RoundedRectangle(width=6, height=3, color=ORANGE)
        rect_1.set_x(-3.3)
        rect_1.set_y(-1.3)

        rect_2 = RoundedRectangle(width=6, height=3, color=ORANGE)
        rect_2.set_x(3.3)
        rect_2.set_y(-1.3)

        rect_1_center = rect_1.get_center()
        continuous_text = Text("Continuous environments", color=ORANGE)
        continuous_text.width = 4
        font_size = continuous_text.font_size
        continuous_text.set_x(rect_1_center[0])
        continuous_text.set_y((rect_1_center[1] + rect_1.height / 2) - 0.35)

        rect_2_center = rect_2.get_center()
        large_text = Text("Large environments", color=ORANGE, font_size=font_size)
        large_text.set_x(rect_2_center[0])
        large_text.set_y((rect_2_center[1] + rect_2.height / 2) - 0.35)

        dq_table = Text("since tables are not used", font_size=32)
        dq_table.set_y(rect_1.get_bottom()[1] - 0.4)

        start = rect_1_center + np.array([-2.5, 0.3, 0])

        arrows_length = 4.1
        discrete_steps = 5

        discrete_arrow = Arrow(start=start, end=start + np.array([arrows_length / discrete_steps, 0, 0]))
        continuous_arrow = Arrow(
            start=start + np.array([0, -1, 0]),
            end=start + np.array([arrows_length / discrete_steps, 0, 0]) + np.array([0, -1, 0])
        )

        nl = NumberLine(length=arrows_length, x_range=[0, 5, 1], include_numbers=True, font_size=20, color=RED_A)
        nl.set_x(start[0] + arrows_length / 2 + 0.23)
        nl.set_y(start[1] - 0.5)

        discrete_text_arrow = Text("Discrete", font_size=16)
        discrete_text_arrow.set_x(nl.get_x())
        discrete_text_arrow.set_y(discrete_arrow.get_y() + 0.3)

        continuous_text_arrow = Text("Continuous", font_size=16)
        continuous_text_arrow.set_x(nl.get_x())
        continuous_text_arrow.set_y(continuous_arrow.get_y() - 0.3)

        def get_length_updater(arrow, should_round=False):
            def updater(num):
                val = arrow.get_length() / arrows_length
                val = val * discrete_steps
                if should_round:
                    val = round(val)
                if discrete_steps - val < 0.1:
                    val = discrete_steps
                num.set_value(val)

            return updater

        discrete_length = DecimalNumber(0, font_size=32)
        discrete_length.set_y(start[1])
        discrete_length.set_x(start[0] + arrows_length + 0.75)
        discrete_length.add_updater(get_length_updater(discrete_arrow, should_round=True))

        continuous_length = DecimalNumber(0, font_size=32)
        continuous_length.set_y(start[1] - 1.05)
        continuous_length.set_x(start[0] + arrows_length + 0.75)
        continuous_length.add_updater(get_length_updater(continuous_arrow))

        def create_table(size_scale):
            start_table_dims = (np.array([5, 3]) * size_scale).astype(int)
            data = np.random.randint(0, 10, size=start_table_dims).astype(str)

            large_table = Table(
                data,
                row_labels=[Text("action") for _ in range(start_table_dims[0])],
                col_labels=[Text("state") for _ in range(start_table_dims[1])],
                include_outer_lines=True,
                line_config={"stroke_width": 0.3},
            )
            large_table.move_to(rect_2)
            large_table.set_y(large_table.get_y()-0.2)
            large_table.scale_to_fit_width(3)
            return large_table

        large_table = create_table(1)

        self.play(
            Write(dq_explained),
        )

        self.wait(4)

        self.play(
            AnimationGroup(
                Write(rect_1),
                Write(rect_2),
                Write(continuous_text),
                Write(large_text),
                Write(discrete_length),
                Write(continuous_length),
                Write(nl),
                Write(dq_table),
                Write(large_table),
                Write(discrete_text_arrow),
                Write(continuous_text_arrow),
                lag_ratio=0.3,
            )
        )

        self.wait(0.5)

        for step in range(discrete_steps):
            distance = ((step + 1) * (arrows_length / discrete_steps))

            self.play(
                GrowArrowCustom(discrete_arrow, distance, rate_func=rate_functions.ease_out_expo, run_time=0.2),
                GrowArrowCustom(continuous_arrow, distance, rate_func=rate_functions.linear),

            )

        self.wait(1)

        for step in range(discrete_steps):
            next_table = create_table(1.5 + step * 0.5)
            self.play(
                ReplacementTransform(large_table, next_table, rate_func=rate_functions.ease_out_expo)
            )
            large_table = next_table

        self.wait(2)

        self.play(
            Unwrite(rect_1),
            Unwrite(rect_2),
            Unwrite(continuous_text),
            Unwrite(large_text),
            Unwrite(discrete_length),
            Unwrite(continuous_length),
            Unwrite(nl),
            Unwrite(dq_table),
            Unwrite(large_table),
            Unwrite(discrete_arrow),
            Unwrite(continuous_arrow),
            Unwrite(dq_explained),
            Unwrite(continuous_text_arrow),
            Unwrite(discrete_text_arrow)
        )

        title7 = Text("Bellman Optimality Equation")
        center = title6.get_left()
        title7.set_x(center[0] + title7.width / 2)
        title7.set_y(center[1])

        updated = Text("In Deep Q Learning, the network is updated by the", font_size=36)
        updated.set_y(2)
        bellman = Text("Bellman Optimality Equation", color="#ff974c")
        bellman.set_y(1.3)

        equation = MathTex(
            r"Q(s, a) = \underset{\text{reward}}{\underbrace{r_{t}}}\;+\ \underset{\text{discounted future rewards}}{\underbrace{\gamma\;\cdot\;\underset{max}{Q}(s_{t+1}, a)}}"
        )
        equation.set_y(-0.8)

        explained = Text("target Q value = reward + discount · max(next Q values)", font_size=36, color=LIGHT_GRAY)
        explained.set_y(-2.5)

        info = Tex(
            r"This means that the optimal current Q value takes into \\ account the reward of the action, and future Q values."
        )
        info2 = Tex(
            r"Choosing the action with the largest Q value therefore\\ maximises future rewards."
        )
        info.set_y(-0.5)
        info2.set_y(-2)

        self.play(
            ReplacementTransform(title6, title7),
        )
        self.wait(1)

        self.play(
            AnimationGroup(
                Write(updated),
                Write(bellman),
                Write(equation, run_time=4),
                Write(explained),
                lag_ratio=0.4
            )
        )

        self.wait(8)

        self.play(
            Unwrite(updated),
            Unwrite(bellman),
            Unwrite(explained),
        )
        self.play(
            AnimationGroup(
                equation.animate.set_y(1.4),
                Write(info),
                lag_ratio=1.2
            )
        )
        self.wait(4.5)
        self.play(
            Write(info2),
        )
        self.wait(4.5)
        self.play(
            AnimationGroup(
                Unwrite(equation),
                Unwrite(info),
                Unwrite(info2),
                lag_ratio=0.3
            )
        )

        title8 = Text("Exploration vs Exploitation")
        center = title7.get_left()
        title8.set_x(center[0] + title8.width / 2)
        title8.set_y(center[1])

        body = Tex(r"The agent may find a policy that receives rewards but\\ isn't optimal, and ", r"exploit", r" (not change) it.",
                   font_size=42)
        body.set_color_by_tex("exploit", ORANGE)
        body.set_y(1.9)

        self.play(
            ReplacementTransform(title7, title8),
        )

        self.wait(1)

        self.play(
            Write(body)
        )

        def pathfinder_animation(scene: Scene, grid_pos, selection_list, final_color: str):
            vals = np.full((7, 7), ".").tolist()

            mobj_vals = []
            for row in vals:
                mobj_row = []
                for v in row:
                    mobj_row.append(Dot(radius=0.04))
                mobj_vals.append(mobj_row)

            mobj_vals[-1][-1] = Star(outer_radius=0.06, color=GREEN)

            bad_grid = MobjectTable(
                mobj_vals,
                include_outer_lines=True,
                h_buff=0.5,
                v_buff=0.5
            )

            bad_grid.set_x(grid_pos[0])
            bad_grid.set_y(grid_pos[1])

            scene.play(
                Write(bad_grid)
            )
            scene.wait(3)
            rects = []
            for box in selection_list:
                bad_grid.add_highlighted_cell((box[0] + 1, box[1] + 1), color=ORANGE)
                entry = bad_grid.get_entries((box[0] + 1, box[1] + 1))

                rect = entry.background_rectangle.copy()
                rect.set_opacity(1)
                rect.scale(1.02)

                rects.append(rect)

                scene.play(
                    FadeIn(rect),
                    run_time=0.15
                )

            scene.play(
                Unwrite(bad_grid)
            )

            scene.play(
                AnimationGroup(
                    *(rect.animate.set_color(final_color) for rect in rects),
                    lag_ratio=0.02
                )
            )

            return rects

        bad_policy = Text("Bad Policy", font_size=30)
        bad_policy.set_y(-1.2)
        bad_policy.set_x(-4)
        self.wait(2)
        self.play(
            Write(bad_policy)
        )

        sl = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 5),
            (2, 5),
            (2, 4),
            (2, 3),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (6, 6)
        ]

        rects = pathfinder_animation(self, (0, -1.2), selection_list=sl, final_color=RED)
        self.wait(2)
        self.play(
            *(ScaleInPlace(r, 0) for r in rects),
            Unwrite(bad_policy)
        )

        for r in rects:
            self.remove(r)

        self.play(
            Unwrite(body)
        )

        explore_text = Tex(r"To prevent this, the agent is sometimes forced to take\\random actions, ", r"exploring",
                           r" more actions.\\This is the greedy epsilon algorithm.", font_size=40)
        explore_text.set_color_by_tex("exploring", ORANGE)
        explore_text.set_y(1.75)

        self.play(
            Write(explore_text)
        )

        self.wait(1)

        good_policy = Text("Good Policy", font_size=30)
        good_policy.set_y(-1.2)
        good_policy.set_x(-4)
        self.play(
            Write(good_policy)
        )

        sl = [
            (0, 0),
            (1, 0),
            (1, 1),
            (2, 1),
            (2, 2),
            (3, 2),
            (3, 3),
            (4, 3),
            (4, 4),
            (5, 4),
            (5, 5),
            (6, 5),
            (6, 6)
        ]

        rects = pathfinder_animation(self, (0, -1.4), selection_list=sl, final_color=GREEN)

        self.wait(2)
        self.play(
            *(ScaleInPlace(r, 0) for r in rects),
            Unwrite(good_policy)
        )

        for r in rects:
            self.remove(r)

        self.play(
            Unwrite(explore_text)
        )

        title9 = Text("Experience Playback")
        center = title8.get_left()
        title9.set_x(center[0] + title9.width / 2)
        title9.set_y(center[1])

        experiences_text = Tex(r"To train the network, a list of all experiences\\ in the episode should be saved.",
                               font_size=55)
        experiences_text.set_y(1.1)

        experiences_text_2 = Tex("Each experience is a tuple of", font_size=55)
        experiences_text_2.set_y(-0.3)

        experience_format = Tex("(state,\ \ action,\ \ reward)", font_size=55, color=ORANGE)
        experience_format.set_y(-1.2)

        why_text = Tex(
            r"This allows the order of experiences to be\\ shuffled, preventing overfitting to an environment.",
            font_size=50)
        why_text.set_y(-2.5)

        self.play(ReplacementTransform(title8, title9))

        self.wait(1)

        self.play(
            AnimationGroup(
                Write(experiences_text),
                AnimationGroup(
                    Write(experiences_text_2),
                    Write(experience_format),
                ),
                Write(why_text),
                lag_ratio=2.4
            )
        )
        self.wait(6)
        self.play(
            Unwrite(why_text),
            Unwrite(experiences_text_2),
            Unwrite(experience_format),
            Unwrite(experiences_text)
        )

        title10 = Text("Applying to a self driving car")
        center = title9.get_left()
        title10.set_x(center[0] + title10.width / 2)
        title10.set_y(center[1])

        actions_text = Text("The possible actions for the agent will be:", t2c={"actions": ORANGE}, font_size=38)
        actions_text.set_y(.8)

        possible_actions = BulletedList(
            "Nothing",
            "Steer Left",
            "Steer Right",
        )

        possible_actions.set_y(-1)
        possible_actions.set_x(0)

        self.play(ReplacementTransform(title9, title10))
        self.wait()

        self.play(
            AnimationGroup(
                Write(actions_text),
                Write(possible_actions),
                run_time=2
            )
        )
        self.wait(5)

        state_text = Text("The state for the agent will be:", t2c={"state": ORANGE}, font_size=38)
        state_text_2 = Text("the distance to the road edge at different places", font_size=38)

        state_text.set_y(1.7)
        state_text_2.set_y(-2.2)

        self.play(
            ReplacementTransform(actions_text, state_text),
            Unwrite(possible_actions),
        )
        self.play(
            Write(state_text_2)
        )
        self.wait(2)
        self.play(
            Unwrite(state_text_2),
            Unwrite(state_text)
        )

        title11 = Text("Hyperparameters")
        center = title10.get_left()
        title11.set_x(center[0] + title11.width / 2)
        title11.set_y(center[1])

        learning_rate = MathTex(r"\alpha\ \ \text{- Learning Rate}", color=ORANGE)
        lr_explained = Tex(
            r"How much the neural network's values\\are changed in gradient descent."
        )
        lr_explained_2 = Tex(
            r"Can cause slow convergence if low, or poor policies if high."
        )
        learning_rate.set_y(1.3)
        lr_explained.set_y(-0.4)
        lr_explained_2.set_y(-1.7)

        self.play(ReplacementTransform(title10, title11))

        self.wait(1)

        self.play(
            Write(learning_rate),
            Write(lr_explained),
            Write(lr_explained_2)
        )
        self.wait(6)
        self.play(
            Unwrite(learning_rate),
            Unwrite(lr_explained),
            Unwrite(lr_explained_2)
        )

        discount_rate = MathTex(r"\gamma\ \ \text{- Discount rate}", color=ORANGE)
        dr_explained = Tex(
            r"How much the agent prefers long term\\rewards over short term rewards."
        )

        discount_rate.set_y(0.8)
        dr_explained.set_y(-0.8)

        self.play(
            Write(discount_rate),
            Write(dr_explained),
        )
        self.wait(5)
        self.play(
            Unwrite(discount_rate),
            Unwrite(dr_explained),
        )

        decay = Tex(r"Epsilon Decay", color=ORANGE)
        decay_explained = MathTex(
            r"\text{How quickly epsilon (}\varepsilon\text{) exponentially decreases.}"
        )
        decay_explained_2 = Tex(
            r"Can find better policies if decay is low,\\ however increases training time."
        )
        decay.set_y(1.3)
        decay_explained.set_y(-0.4)
        decay_explained_2.set_y(-1.7)

        self.play(
            Write(decay),
            Write(decay_explained),
            Write(decay_explained_2)
        )
        self.wait(6.5)
        self.play(
            Unwrite(decay),
            Unwrite(decay_explained),
            Unwrite(decay_explained_2),
            Unwrite(title11)
        )
        self.wait(1)

        coding = Tex(r"Check out my next video, about\\coding a Deep Q Network in Python!")
        self.play(
            Write(coding)
        )
        self.wait(5)
        self.play(
            Unwrite(coding)
        )


if __name__ == "__main__":
    os.system("pipenv run manim render screens.py -qp -t")
