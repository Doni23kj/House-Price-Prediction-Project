import pickle
import pandas as pd


def get_number(prompt):
    while True:
        user_input = input(prompt).strip()

        try:
            value = float(user_input)
            if value < 0:
                print("Ошибка: введите число больше или равно 0.")
                continue
            return value
        except ValueError:
            print("Ошибка: нужно ввести только число!")


with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

print("Введите данные для предсказания")

income = get_number("Avg Area Income: ")
age = get_number("Avg Area House Age: ")
rooms = get_number("Avg Area Number of Rooms: ")
while True:
    bedrooms = get_number("Avg Area Number of Bedrooms: ")
    if bedrooms > rooms:
        print("Ошибка: bedrooms не может быть больше rooms.")
    else:
        break

population = get_number("Area Population: ")

data = pd.DataFrame([{
    "Avg. Area Income": income,
    "Avg. Area House Age": age,
    "Avg. Area Number of Rooms": rooms,
    "Avg. Area Number of Bedrooms": bedrooms,
    "Area Population": population
}])

prediction = model.predict(data)[0]

if prediction < 0:
    prediction = 0

print("Predicted house price:", round(prediction, 2))