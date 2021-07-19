from sdv.demo import load_tabular_demo
from sdv.tabular import CTGAN

data = load_tabular_demo('student_placements')

print(data.head())

model = CTGAN()
model.fit(data)

new_data = model.sample(200)

print(new_data.head())
