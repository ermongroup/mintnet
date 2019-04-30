import math
import numpy as np

def toy1(partition=10, radius = 15, size=1000):
    std = [[1., 0.], [0., 1.]]
    assert(partition >= 1)
    samples = []
    for i in range(partition):
        mean = [radius * math.cos(i*(2*math.pi/partition)), radius * math.sin(i*(2.*math.pi/partition))]
        samples.extend(np.random.multivariate_normal(mean, std, size))
    np.random.shuffle(samples)
    return samples


def toy2(num=5, radius=5, interval=5, size=int(1e5), width=2):
    samples = []
    mean = [0., 0.]
    std = [[0.1, 0], [0., 0.1]]
    samples.extend(np.random.multivariate_normal(mean, std, int(size / 100.)))

    def sample_cir(num, radius, interval, size):
        prob_circle = []
        for i in range(num):
            prob_circle.append(1. / ((radius + i * interval) ** 2))
        prob_circle = np.array(prob_circle)
        prob_circle /= prob_circle.sum()
        prob_circle = prob_circle.tolist()
        choices = np.random.choice(num, size, prob_circle)
        return choices

    choices = sample_cir(num, radius, interval, int(size * 99. / 100.))
    for c in choices:
        theta = np.random.uniform(0, 2 * math.pi, size=1)[0]
        r = np.random.uniform(0, width, size=1)[0]
        [x, y] = [(radius + c * interval + r) * math.cos(theta), (radius + c * interval + r) * math.sin(theta)]
        samples.extend([[x, y]])
    np.random.shuffle(samples)
    return samples


def toy3(size=10000):
    std = [[21., 0.], [0., -1.]]
    mean = [0, 0]
    return np.random.multivariate_normal(mean, std, size)

