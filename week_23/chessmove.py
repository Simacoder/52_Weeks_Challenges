from collections import deque

# Class to represent a square on the chessboard
class Square:
    def __init__(self, x_coord, y_coord, parent=None):
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.parent = parent
        self.children = []

    def generate_legal_moves(self):
        # All 8 possible knight moves (in L-shape)
        row_moves = [-2, -1, 1, 2, 2, 1, -1, -2]
        col_moves = [1, 2, 2, 1, -1, -2, -2, -1]

        for dx, dy in zip(row_moves, col_moves):
            nx, ny = self.x_coord + dx, self.y_coord + dy

            # Only add valid board positions (0–7 for an 8x8 board)
            if 0 <= nx < 8 and 0 <= ny < 8:
                self.children.append(Square(nx, ny, self))

# BFS function to find the shortest path from start to end square
def knight_travails(start_coords, end_coords):
    start_x, start_y = start_coords
    queue = deque([Square(start_x, start_y)])
    visited = set()
    end_coords = tuple(end_coords)

    while queue:
        current = queue.popleft()
        coords = (current.x_coord, current.y_coord)

        if coords == end_coords:
            return current

        if coords in visited:
            continue

        visited.add(coords)

        current.generate_legal_moves()
        queue.extend(current.children)

def construct_path(start, end):
    end_square = knight_travails(start, end)
    path = []

    while end_square:
        path.append((end_square.x_coord, end_square.y_coord))
        end_square = end_square.parent

    return path[::-1]

# Example usage
print(construct_path([3, 4], [0, 1]))
