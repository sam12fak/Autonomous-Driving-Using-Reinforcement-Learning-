from manim import *
from map_grid import MapGrid
from misc_funcs import *
from roads import circle_collision_road
from typing import Optional, Tuple, List, Iterable


ROAD_MAP = [(1, 1),(2, 1),(3, 1),(4, 1),(5, 1),(6, 1),(7, 1),(8, 1),(9, 1),(10, 1),(11, 1),(12, 1),(13, 1),(14, 1),(14, 2),(14, 3),(14, 4),(13, 4),(12, 4),(11, 4),(10, 4),(10, 5),(10, 6),(10, 7),(9, 7),(8, 7),(7, 7),(6, 7),(5, 7),(4, 7),(3, 7),(2, 7),(1, 7),(1, 6),(1, 5),(1, 4),(1, 3),(1, 2)]


class Intro(Scene):
    def construct(self):
        title = Text("Creating a self-driving car simulation", font_size=36)
        title.set_y(0.5)

        p3 = Text("Part 3", color=ORANGE, font_size=24)
        p3.next_to(title, DOWN)

        self.play(
            Write(title)
        )

        self.wait()

        self.play(
            Write(p3)
        )

        self.wait(3)

        self.play(
            Unwrite(p3),
            Unwrite(title)
        )


class MapCreationAndCollision(Scene):
    def construct(self):
        mapgrid = MapGrid(self, (16, 9), grid_size=0.63, initial_pos=(0, -0.5))
        road_map_matrix = coordinates_to_matrix(ROAD_MAP, (16, 9))

        drawn_desc = line_centerer("The first step to creating the map is drawing road lines.", font_size=35)
        self.wait()
        drawn_desc.next_to(mapgrid.table, UP, 0.5)
        self.play(Write(drawn_desc))

        drawn_road_cells = mapgrid.draw_roads(road_map_matrix, animation_time=1)
        self.wait(2)

        self.play(Unwrite(drawn_desc))

        self.wait(0.5)

        generation_desc = line_centerer("By checking for preset patterns, roads can be added.", font_size=35)
        generation_desc.next_to(mapgrid.table, UP, 0.5)
        self.play(Write(generation_desc))
        self.wait(1.5)

        mapgrid.generate_roads(road_map_matrix, animate=True, pattern_loc=(-6, 1))
        mapgrid.remove_cells(drawn_road_cells)

        self.wait(2)

        self.play(Unwrite(generation_desc))

        mapgrid.move_to((0, 10))

        collision_text = line_centerer("For ray tracing and road collision, we need to know if a point is on a road.")
        self.play(Write(collision_text))
        self.wait(5)
        self.play(Unwrite(collision_text))

        mapgrid.move_to((0, -0.5))
        straight_road = Text("Straight Road Collision", font_size=37)
        straight_road.next_to(mapgrid.table, UP, 0.5)

        self.play(Write(straight_road))

        subgrid = mapgrid.focus_on_subgrid((5, 0), (5, 2), focus_loc=(0, -1), max_size=6)

        straight_road_point = Circle(0.04, ORANGE, fill_opacity=1)
        straight_road_point.set_z_index(2)  # above road
        straight_road_point.move_to((0, -1, 0))

        back_grid = MobjectTable(
            [[Square(2 if (r==1 and c==1) else 1, stroke_opacity=0) for r in range(3)] for c in range(3)],
            v_buff=0,
            h_buff=0
        )
        back_grid.move_to((0, -1, 0))

        straight_collision = line_centerer("If a point is on a straight road grid space, it is always overlapping.")
        straight_collision.set_y(2)
        self.play(Write(straight_collision), Write(straight_road_point), Write(back_grid))
        self.wait(2)

        current_pos = straight_road_point.get_center()
        for _ in range(5):
            self.play(
                straight_road_point.animate.move_to(random_offset_position(tuple(current_pos), 2.8, 2.8, cubic_s_distribution))  # function causes more points at edge
            )

        self.play(
            Unwrite(back_grid),
            Unwrite(straight_collision),
            Unwrite(straight_road_point),
            Unwrite(straight_road)
        )

        mapgrid.unfocus_subgrid(subgrid, (5, 0), (5, 2))

        curved_road = Text("Curved Road Collision", font_size=37)
        curved_road.next_to(mapgrid.table, UP, 0.5)

        self.play(Write(curved_road))

        subgrid = mapgrid.focus_on_subgrid((4, 10), (5, 11), focus_loc=(0, -1), max_size=4)

        back_grid = MobjectTable(
            [[Square(2, color=BLACK).set_opacity(0) for _ in range(2)] for _ in range(2)],
            line_config={"color": WHITE},
            v_buff=0,
            h_buff=0
        ).set_z_index(2)
        back_grid.move_to(subgrid.get_center())

        curved_collision = line_centerer("Circles can be used to calculate collision on curved roads.")
        curved_collision.set_y(2)

        self.play(Write(curved_collision), Write(back_grid))

        self.wait(2)

        nl_config = {"include_numbers": True}
        axes = Axes(x_range=[0, 2.01, 1], y_range=[0, 2.01, 1], x_length=4.1, y_length=4.1, axis_config=nl_config, tips=False).set_z_index(2)
        axes.move_to(subgrid.get_corner(DOWN+LEFT)-np.array([0.25, 0.25, 0]) + np.array([2.1, 2.1, 0]))

        self.play(Write(axes))

        lr_corner = subgrid.get_corner(DOWN+RIGHT)
        less_than_arc = Arc(radius=2, angle=TAU/4, arc_center=lr_corner, fill_color=ORANGE, stroke_color=BLUE, stroke_width=6).set_z_index(2)
        less_than_arc.rotate(0.5*PI, about_point=lr_corner)

        greater_than_arc = Arc(radius=4, angle=TAU/4, arc_center=lr_corner, stroke_color=BLUE, stroke_width=6).set_z_index(2)
        greater_than_arc.rotate(0.5*PI, about_point=lr_corner)

        self.play(Write(less_than_arc), Write(greater_than_arc))
        self.wait(1)

        greater_than_arc.add_line_to(lr_corner)
        less_than_arc.add_line_to(lr_corner)

        greater_than_arc_box = Square(side_length=4).move_to(subgrid.get_center()).set_z_index(2)
        greater_than_difference = Difference(greater_than_arc_box, greater_than_arc, color=YELLOW, fill_opacity=1)

        less_than_arc.generate_target(True)
        less_than_arc.target.set_fill(opacity=1)
        less_than_arc.target.set_stroke(opacity=0)

        self.play(Write(greater_than_difference), Unwrite(greater_than_arc), MoveToTarget(less_than_arc))

        circle_less_than = Text("Less than this", font_size=20, color=BLACK)\
            .move_to(greater_than_difference.get_center_of_mass()).set_z_index(3)\
            .shift([-0.6, 0.8, 0])
        circle_greater_than = Text("Greater than this", font_size=14, color=BLACK)\
            .move_to(less_than_arc.get_center_of_mass()).set_z_index(3)\
            .shift([0.4, -0.2, 0])

        self.play(Write(circle_less_than), Write(circle_greater_than))
        self.wait(1)

        collision_point = Circle(0.1, ORANGE, fill_opacity=1, stroke_opacity=0).set_z_index(4)
        collision_point.move_to(subgrid.get_center())
        start = collision_point.get_center()
        for n in range(8):
            new = random_offset_position(tuple(start), 4, 4, cubic_s_distribution)
            diff = np.array(new) - np.array(start)

            collision_point.generate_target(True)
            collision_point.target.move_to(new)

            above_x = bool(diff[1] > 0)
            right_y = bool(diff[0] > 0)

            distance_into_box = (abs(np.array([-2, 2]) - np.array(diff[:2]))) % 2

            overlap = circle_collision_road(
                width=2,
                grid_size=2,
                rotation=0,
                grid_location=(int(right_y), int(not above_x)),
                relative_point=distance_into_box
            )

            collision_point.target.set_fill(color=GREEN if overlap else RED)

            self.play(MoveToTarget(collision_point))

        self.wait(1)

        self.play(
            Unwrite(collision_point),
            Unwrite(circle_less_than),
            Unwrite(circle_greater_than),
            Unwrite(less_than_arc),
            Unwrite(greater_than_difference),
            Unwrite(axes),
            Unwrite(curved_road),
            Unwrite(back_grid),
            Unwrite(curved_collision)
        )

        self.play(
            *(obj.animate.set_opacity(0) for obj in subgrid.submobjects)
        )

        mapgrid.unfocus_subgrid(subgrid, (4, 10), (5, 11), animate_in=False)

        return mapgrid


class MovementExplanation(Scene):
    def construct(self):
        moving_car = Text("Moving the car")
        moving_car.move_to((0, 3, 0))

        back_rect = RoundedRectangle(0.05, fill_opacity=1, fill_color=GREEN, stroke_opacity=0)
        back_rect.width = 10
        back_rect.height = 5.67
        back_rect.move_to((0, -0.5, 0))

        self.play(Write(moving_car), Write(back_rect))

        movement_explained = line_centerer("To find the updated car location, we need the velocity and angle.", width=13, color=BLACK)
        movement_explained.next_to(back_rect.get_center(), UP, 0.5)

        self.wait(1.5)
        self.play(Write(movement_explained))

        movement_explained_2 = line_centerer("We can use this to create a 2D vector for the change in location.", width=13, color=BLACK)
        movement_explained_2.next_to(back_rect.get_center(), DOWN, 0.5)
        self.wait(1.5)
        self.play(Write(movement_explained_2))

        self.wait(4)

        self.play(Unwrite(movement_explained), Unwrite(movement_explained_2))

        velocity_value = 3
        velocity = Text(f"Velocity: {velocity_value}", font_size=24, color=BLACK).move_to((-4.5, 1.8, 0))

        angle_value = 0
        angle = Text(f"Angle: {angle_value}", font_size=24, color=BLACK)
        angle.next_to(velocity, DOWN, 0.2)

        movement_vector = Arrow(start=(0, -1, 0), end=back_rect.get_center()+UP*2.5, color=BLACK)
        vector = np.array([0, velocity_value])
        text_left_pos = ((np.array(movement_vector.get_start()) + np.array(movement_vector.get_end())) / 2) + RIGHT * 0.3

        vector_text = Text(f"({vector[0]}, {vector[1]})", font_size=24, color=BLACK)
        vector_text.move_to(text_left_pos, aligned_edge=LEFT)

        self.play(
            Write(movement_vector),
            Write(vector_text),
            Write(velocity),
            Write(angle)
        )

        vertical_line = Line(start=(0, -0.75, 0), end=(0, 1.75, 0), color=BLACK, stroke_width=1, buff=0)
        angle_arc = Arc(0.4, angle=0, arc_center=(0, -0.75, 0), color=BLACK).rotate(PI/2, about_point=(0, -0.75, 0))

        def get_angle_animations(target_angle, current_angle, angle_arc, angle_text, vector, vector_annotation, annotation_offset, vector_annotation_override=None):
            at_target = Text(f"Angle: {round(target_angle*(180/PI), 0)}", font_size=angle_text.font_size, color=BLACK).move_to(angle_text.get_center())

            vector_target = vector.generate_target(True)
            vector_target.rotate(2*PI - (target_angle-current_angle), about_point=vector_target.get_start())

            new_vector = np.array(vector_target.get_end()) - np.array(vector_target.get_start()) if vector_annotation_override is None else vector_annotation_override
            new_vector = np.round((new_vector / np.linalg.norm(new_vector))*velocity_value, 1)  # scale to length
            vector_anno_target = Text(f"({new_vector[0]}, {new_vector[1]})", font_size=vector_annotation.font_size, color=BLACK).move_to(vector_annotation.get_center())

            text_left_pos = ((np.array(vector_target.get_start()) + np.array(vector_target.get_end())) / 2) + annotation_offset
            vector_anno_target.move_to(text_left_pos)

            arc_target = Arc(0.4, angle=target_angle, arc_center=(0, -0.75, 0), color=BLACK).rotate(PI/2-target_angle, about_point=(0, -0.75, 0))

            return (
                Transform(angle_text, at_target),
                Transform(vector, vector_target),
                Transform(vector_annotation, vector_anno_target),
                Transform(angle_arc, arc_target),
                    )

        explanation = line_centerer("At 0 angle, the change is simply a vector of (0, velocity)", width=18, font_size=30, color=BLACK)
        explanation.move_to((0, -2.5, 0))

        self.play(Write(explanation))

        self.wait(3)

        explanation2 = line_centerer("At an angle, the change is this vector, rotated", width=18, font_size=30, color=BLACK)
        explanation2.move_to((0, -2.5, 0))

        self.play(
            Unwrite(explanation),
        )
        self.play(
            Write(explanation2)
        )

        self.wait(2.5)

        self.play(
            *get_angle_animations(PI/4, 0, angle_arc, angle, movement_vector, vector_text, RIGHT+DOWN * 0.5),
            Write(vertical_line),
        )

        self.wait()

        self.play(
            *get_angle_animations(1.7*PI, PI/4, angle_arc, angle, movement_vector, vector_text, DOWN+(LEFT*0.5) * 0.5),
        )

        self.wait()

        self.play(
            *get_angle_animations(PI*0.6, 1.7*PI, angle_arc, angle, movement_vector, vector_text, LEFT+DOWN * 0.5),
        )

        self.wait()

        self.play(
            Unwrite(angle),
            Unwrite(angle_arc),
            Unwrite(movement_vector),
            Unwrite(vector_text),
            Unwrite(vertical_line),
            Unwrite(velocity),
            Unwrite(explanation2)
        )

        turning = Text("Rotating Car").move_to(moving_car)

        self.play(
            Transform(moving_car, turning)
        )

        self.wait(1.3)

        facts = line_centerer("To calculate the rotational change of the car, based on the wheel angle, we find the radius of the turning circle.", color=BLACK).set_y(-0.5)
        self.play(Write(facts))
        self.wait(3)
        self.play(Unwrite(facts))

        turning_circle = Circle(2, color=BLACK).move_to(back_rect.get_center())
        radius = Line(start=turning_circle.get_center(), end=turning_circle.get_center()+np.array([-2, 0, 0]), color=BLACK)
        radius_label = Text("r", color=BLACK, font_size=30).next_to(radius.get_center(), DOWN, 0.2)

        car = ImageMobject("assets/car.png").scale(0.1).rotate(PI)
        car.move_to(radius.get_end())

        self.play(Write(turning_circle), Write(radius), FadeIn(car), Write(radius_label))

        global car_rot
        car_rot = 0
        car_vel = 0.05

        def move_car(car):
            global car_rot
            car_shift = rotate_vector_acw(np.array((0, car_vel)), -car_rot)
            angular_velocity = car_vel / 2
            car.shift((car_shift[0], car_shift[1], 0))
            car.rotate(PI*2-angular_velocity)
            car_rot += angular_velocity*59  # 59 seems to work not sure why?

        self.play(UpdateFromFunc(car, move_car), run_time=2)
        self.play(
            Unwrite(turning_circle),
            Unwrite(radius),
            FadeOut(car),
            Unwrite(radius_label)
        )

        text_center = (0, -0.5, 0)
        text1 = line_centerer("The radius can be calculated using the wheel's steering angle.", color=BLACK, width=19, font_size=26).next_to(text_center, UP, 1)
        text2 = line_centerer("The 4 wheels can be modelled as 2 sets, the front and the back.", color=BLACK, width=19, font_size=26).move_to(text_center)
        text3 = line_centerer("If perpendicular lines are drawn from each wheel, they intersect at the center of the turning circle.", color=BLACK, font_size=26).next_to(text_center, DOWN, 1)
        self.play(
            AnimationGroup(
                Write(text1),
                Write(text2),
                Write(text3),
                lag_ratio=3,
                run_time=10
            )
        )
        self.wait(6.5)
        self.play(
            AnimationGroup(
                Unwrite(text1),
                Unwrite(text2),
                Unwrite(text3),
                lag_ratio=0.3,
                run_time=1
            )
        )

        tire_distance = 0.9

        tire_steer = ImageMobject("assets/tread.png").scale(0.15).set_z_index(1)
        tire_back = tire_steer.copy()

        tire_steer.move_to((-2.5, 0, 0)).rotate(2*PI-PI/6, about_point=tire_steer.get_center())
        tire_back.next_to(tire_steer, DOWN, tire_distance)

        wheel_vertical_line = Line(start=tire_steer.get_center(), end=tire_back.get_center(), color=BLACK)
        self.play(
            Write(wheel_vertical_line),
            FadeIn(tire_steer),
            FadeIn(tire_back),
        )

        front_tire_label = Text("Front tyres", font_size=24, color=BLACK).next_to(tire_steer.get_center(), LEFT, 0.2)
        back_tire_label = Text("Back tyres", font_size=24, color=BLACK).next_to(tire_back.get_center(), LEFT, 0.2)

        self.play(
            Write(front_tire_label),
            Write(back_tire_label)
        )

        self.wait(2)

        self.play(
            Unwrite(front_tire_label),
            Unwrite(back_tire_label)
        )

        horizontal_tri_length = -tire_distance*np.tan(PI/2- (2*PI-PI/6))
        circle_center = tire_back.get_center()+np.array([horizontal_tri_length, 0, 0])
        horizontal_triangle_line = Line(start=tire_back.get_center(), end=circle_center, color=BLACK)

        radius_triangle_line = Line(start=tire_steer.get_center(), end=circle_center, color=BLACK)
        radius_label = MathTex("r", font_size=30, color=BLACK).next_to(radius_triangle_line.get_center(), UP+RIGHT, 0.05)

        turning_circle = Circle(radius_triangle_line.get_length(), color=GREY).move_to(circle_center)

        self.play(
            Write(horizontal_triangle_line),
            Write(radius_triangle_line),

        )

        self.wait()

        self.play(
            Write(radius_label)
        )

        self.wait()

        radius_label.generate_target()
        radius_label.target.set_color(GREY)

        radius_triangle_line.generate_target()
        radius_triangle_line.target.set_color(GREY)

        self.play(Write(turning_circle), MoveToTarget(radius_label), MoveToTarget(radius_triangle_line))

        radius_label.generate_target()
        radius_label.target.set_color(BLACK)

        radius_triangle_line.generate_target()
        radius_triangle_line.target.set_color(BLACK)

        self.wait(2)

        self.play(Unwrite(turning_circle), MoveToTarget(radius_label), MoveToTarget(radius_triangle_line))

        vertical_dotted = DashedLine(start=wheel_vertical_line.get_start(), end=wheel_vertical_line.get_start()+np.array((0, 1, 0)), color=GREY)
        angled_dotted = DashedLine(start=wheel_vertical_line.get_start()+np.array((0, 1, 0)), end=tire_back.get_center(), color=GREY)
        angled_dotted.rotate(2*PI - PI/6, about_point=tire_steer.get_center())

        self.play(
            Write(vertical_dotted),
            Write(angled_dotted)
        )

        turning_angle_key = MathTex(r"\theta = Wheel\,angle", font_size=36, color=BLACK).next_to(horizontal_triangle_line, DOWN, 0.5)
        self.play(
            Write(turning_angle_key)
        )

        self.wait(1.5)

        turning_angle = Angle(vertical_dotted, angled_dotted, 0.6, quadrant=(1, -1), other_angle=True, color=BLACK)
        turning_angle_label = MathTex(r"\theta", color=BLACK, font_size=36).next_to(turning_angle.get_center(), UP+(RIGHT*0.2), 0.15)

        self.play(
            Write(turning_angle),
            Write(turning_angle_label)
            )

        self.wait(0.5)

        turning_angle_other = Angle(wheel_vertical_line, angled_dotted, 0.6, color=BLACK, other_angle=True)
        turning_angle_other_label = turning_angle_label.copy().next_to(turning_angle_other.get_center(), DOWN+(LEFT*0.2), 0.15)

        self.play(
            Write(turning_angle_other),
            Write(turning_angle_other_label)
        )

        self.wait(1)

        right_angle = RightAngle(radius_triangle_line, angled_dotted, color=GREEN_A, length=0.3)
        self.play(
            Write(right_angle)
        )

        interior_angle = Angle(radius_triangle_line, wheel_vertical_line, 0.6, color=BLACK, other_angle=True)
        interior_angle_label = MathTex(r"90 - \theta", font_size=16, color=BLACK).next_to(right_angle.get_center(), DOWN+(RIGHT*0.2), 0.3)

        self.wait(1)

        self.play(
            Write(interior_angle),
            Write(interior_angle_label)
        )

        self.wait(2)

        self.play(
            Unwrite(right_angle),
            Unwrite(turning_angle_other),
            Unwrite(turning_angle_other_label),
        )

        wheel_distance = MathTex("x", font_size=32, color=BLACK).next_to(wheel_vertical_line, LEFT, 0.1)
        wheel_distance_key = MathTex("x = Wheel\,distance", font_size=36, color=BLACK).next_to(turning_angle_key, DOWN, 0.3)

        self.play(
            Write(wheel_distance),
            Write(wheel_distance_key)
        )

        self.wait(2)

        trig1 = MathTex(r"cos(90-\theta) = ", r"{x", r"\over", r"r}", color=BLACK)
        trig1.move_to((2.3, 0.7, 0))

        trig1_explained = Text("Using trigonometry", font_size=24, color=BLACK).next_to(trig1, UP, 0.15)

        self.play(
            Write(trig1),
            Write(trig1_explained)
        )

        self.wait(2)

        trig2_copy = trig1.copy()
        trig2 = MathTex(r"sin(\theta) = ", r"{x", r"\over", r"r}", color=BLACK).next_to(trig1, DOWN, 0.3)

        self.play(
            Transform(trig2_copy, trig2),
        )
        self.remove(trig2)

        self.wait(2)

        trig3_copy = trig2.copy()
        trig3 = MathTex(r"r = {x \over sin(\theta)}", color=BLACK).next_to(trig2, DOWN, 0.3)
        self.play(
            Transform(trig3_copy, trig3)
        )
        self.remove(trig3)

        self.wait(2)

        trig3_copy.generate_target()
        trig3_copy.target.move_to((2.3, -0.5, 0))

        self.play(
            Unwrite(trig1),
            Unwrite(trig2_copy),
            Unwrite(interior_angle),
            Unwrite(interior_angle_label),
            Unwrite(trig1_explained),
        )

        self.play(
            MoveToTarget(trig3_copy),
        )

        self.wait(2)

        self.play(
            Unwrite(turning_angle_key),
            Unwrite(wheel_distance_key),
            Unwrite(angled_dotted),
            Unwrite(vertical_dotted),
            Unwrite(wheel_vertical_line),
            Unwrite(wheel_distance),
            Unwrite(radius_triangle_line),
            Unwrite(radius_label),
            Unwrite(horizontal_triangle_line),
            Unwrite(trig3_copy),
            Unwrite(turning_angle),
            Unwrite(turning_angle_label),
            FadeOut(tire_back),
            FadeOut(tire_steer)
        )

        angular_velocity = line_centerer("The angular velocity (change in rotation) can be found using the radius of the turning circle.", color=BLACK)
        angular_velocity.move_to((0, -0.5, 0))

        self.play(
            Write(angular_velocity)
        )

        angular_velocity.generate_target()
        angular_velocity.target.move_to((0, 1, 0))

        angular_velocity_equation = MathTex(r"w = {v \over r}", color=BLACK).move_to((0, -0.5, 0))
        explained = line_centerer("Angular velocity is linear velocity divided by radius.", color=BLACK).move_to((0, -2, 0))

        self.play(
            MoveToTarget(angular_velocity),
            Write(angular_velocity_equation),
            Write(explained)
        )

        self.wait(3)

        self.play(
            Unwrite(angular_velocity),
            Unwrite(angular_velocity_equation),
            Unwrite(explained),
            Unwrite(back_rect)
        )

        final = line_centerer("Now the car's rotation and location can be updated using the steering angle and velocity.").move_to((0, -0.5, 0))
        self.play(
            Write(final)
        )
        self.wait(3)
        self.play(
            Unwrite(final)
        )


class AllScene(Scene):
    def construct(self):
        Intro.construct(self)
        mapgrid = MapCreationAndCollision.construct(self)
        MovementExplanation.construct(self)


if __name__ == "__main__":
    os.system("pipenv run manim scenes.py -qm")
