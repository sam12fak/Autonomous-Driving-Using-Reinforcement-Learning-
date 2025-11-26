from manim import *
from map_grid import MapGrid
from roads import *
import os
from misc_funcs import coordinates_to_matrix, random_offset_position
from scenes import ROAD_MAP

os.environ["PATH"] = os.environ["PATH"] + r";C:\Users\Heidi\Desktop\epq\FFMPEG\bin\\"


def prepare_test_mapgrid(scene):
    mapgrid = MapGrid(scene, dimensions=(16, 9))

    road_matrix = coordinates_to_matrix(ROAD_MAP, (16, 9))

    # not animating so dont bother adding drawn roads
    """
    mapgrid.draw_roads(
        road_matrix,
        animation_time=None
    )
    """

    mapgrid.generate_roads(road_matrix, animate=False)
    return mapgrid


class MapTest(Scene):
    def construct(self):
        mapgrid = MapGrid(self, dimensions=(16, 9))

        self.wait(1)

        road_matrix = coordinates_to_matrix(ROAD_MAP, (16, 9))

        cells = mapgrid.draw_roads(
            road_matrix,
            animation_time=None
        )

        mapgrid.generate_roads(road_matrix, animate=False)

        mapgrid.remove_cells(cells)


class RoadTest(Scene):
    """
    ROTATING ROAD TEST
    """
    def construct(self):
        road1 = CurvedRoad(0, 1, 0).mob
        road2 = CurvedRoad(90, 1, 0).mob
        road3 = CurvedRoad(180, 1, 0).mob
        road4 = CurvedRoad(270, 1, 0).mob

        group = Group(road1, road2, road3, road4)

        group.arrange(RIGHT)

        self.add(group)


class StraightRoadCollisionTest(Scene):
    def construct(self):
        road = StraightRoad(0, 1)
        self.add(road.mob)

        road_pos = (road.mob.get_x(), road.mob.get_y(), 0)

        collision_point = Circle(0.05, ORANGE, fill_opacity=1)
        size = road.grid_size

        self.add(collision_point)

        for n in range(10):
            self.play(
                collision_point.animate.move_to(random_offset_position(road_pos, size, size)),
                run_time=0.1
            )


class GridFocusTest(Scene):
    def construct(self):
        mapgrid = prepare_test_mapgrid(self)
        self.wait(1)
        subgrid = mapgrid.focus_on_subgrid((1, 1), (5, 5))
        self.wait(3)
        mapgrid.unfocus_subgrid(subgrid, (1, 1), (5, 5))


class CurvedRoadCollisionTest(Scene):
    def construct(self):
        curved_roads = [CurvedRoad(0, 1, n) for n in range(4)]


if __name__ == "__main__":
    os.system("pipenv run manim test_scenes.py -qm")
