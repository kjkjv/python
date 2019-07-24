class Fridge:
    def __init__(self):
        self.isOpened = False
        self.foods = []

    def open(self):
        self.isOpened = True
        print('냉장고 문을 열었어요')

    def put(self, thing):
        if self.isOpened:
            self.foods.append(thing)
            print('냉장고 속에 음식이 들어갔네')
        else:
            print('냉장고 문이 닫혔음')

    def close(self):
        self.isOpened = False
        print('냉장고 문을 닫음')


class Food:
    pass

