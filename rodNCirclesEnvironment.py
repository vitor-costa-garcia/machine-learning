import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

	def generate_circle(precision: int):
		y = np.sin(np.linspace(0, 2*np.pi, precision))
		X = np.cos(np.linspace(0, 2*np.pi, precision))
		return np.array([X, y]).T

	def get_q(circle, q:int) -> (np.array, np.array):
		precision = circle.shape[0]
		return circle[int(precision/4)*(q-1):int(precision/4)*q]

	circle = generate_circle(1000)

	circle_1q = get_q(circle, 1)
	circle_2q = get_q(circle, 2)
	circle_3q = get_q(circle, 3)
	circle_4q = get_q(circle, 4)

	fig, ax = plt.subplots(1, 1, figsize=(10, 10))

	ax.plot(circle_1q[:, 0] - 2, circle_1q[:, 1] - 2, c='red')
	ax.plot(circle_2q[:, 0] + 2, circle_2q[:, 1] - 2, c='blue')
	ax.plot(circle_3q[:, 0] + 2, circle_3q[:, 1] + 2, c='yellow')
	ax.plot(circle_4q[:, 0] - 2, circle_4q[:, 1] + 2, c='green')
	ax.plot(circle[:, 0]*1.5, circle[:, 1]*1.5, c='purple')

	#Reward points
	rew = generate_circle(20)
	rew *= 1.7

	limits = np.array([
		[2, 2],
		[-2, 2],
		[-2, -2],
		[2, -2],
		[2, 2],
	])

	agent = np.array([[ -1.8, np.random.normal(0, 0.1, 1)[0]]])

	ax.plot(limits[:, 0], limits[:, 1])
	ax.scatter(rew[:, 0], rew[:, 1], marker='*', c='yellow', edgecolors='black', s=80)
	ax.scatter(agent[:, 0], agent[:, 1], marker='s', c='brown')

	plt.show()