from game_controller import GameController

if __name__ == "__main__":
    game = GameController()
    game.startGame()
    while True:
        game.update()