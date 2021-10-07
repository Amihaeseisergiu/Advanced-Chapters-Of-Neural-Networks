import numpy as np
import plotly.graph_objects as go
from numpy.random import uniform as runi

def plot(solutions, points):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            mode='markers',
            name='-1',
            x=points[0:50, 0],
            y=points[0:50, 1],
            marker=dict(
                color='green',
                size=10,
                line=dict(
                    color='black',
                    width=2
                )
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            name='1',
            x=points[50:, 0],
            y=points[50:, 1],
            marker=dict(
                color='blue',
                size=10,
                line=dict(
                    color='black',
                    width=2
                )
            )
        )
    )

    for solution in solutions:
        if solution[0] == 0 and solution[1] == 0:
            print("Can't plot")
            return None

        x = None
        y = None
        if solution[0] == 0:
            x = np.linspace(0, 100)
            y = (-solution[2] - solution[0] * x)/(solution[1])
        elif solution[1] == 0:
            y = np.linspace(0, 100)
            x = (-solution[2] - solution[1] * y)/(solution[0])
        else:
            x = np.linspace(0, 100)
            y = (-solution[2] - solution[0] * x)/(solution[1])

        fig.add_trace(
            go.Scatter(x=x, y=y, 
                mode='lines', name='line')
        )
    
    fig.data[0].visible = True

    steps = []
    for i in range(2, len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": "Plot line equation" \
                        f" {solutions[i-2,0]}x {'' if solutions[i-2,1] < 0 else '+'}" \
                        f" {solutions[i-2,1]}y {'' if solutions[i-2,2] < 0 else '+'}" \
                        f" {solutions[i-2,2]} = 0 <br>" \
                        f"Correctly classified: {quality(solutions[i-2], points)}"
                }],
        )
        step["args"][0]["visible"][0] = True
        step["args"][0]["visible"][1] = True
        step["args"][0]["visible"][i] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Current step: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )

    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, 100])
    fig.show()

points = np.concatenate((
    np.concatenate((runi([0, 0], [45, 100], size=(50, 2)), np.full((50, 2), [1,-1])), axis=1),
    np.concatenate((runi([55, 0], [100, 100], size=(50, 2)), np.ones((50, 2))), axis=1)
))
solution = np.array([runi(-100, 100), runi(-100, 100), runi(0, 100)])

def classify(line, point):
    if line[0] != 0:
        if line[0] > 0:
            return point[3] == np.sign(np.dot(line, point[0:3]))
        else:
            return point[3] == -np.sign(np.dot(line, point[0:3]))
    elif line[0] == 0 and line[1] != 0:
        if line[1] > 0:
            return point[3] == np.sign(np.dot(line, point[0:3]))
        else:
            return point[3] == -np.sign(np.dot(line, point[0:3]))
    else:
        return False

def candidate(line, step):
    return line + np.random.standard_normal(3) * step
    
def quality(candidate, points):
    right = 0

    for point in points:
        if classify(candidate, point):
            right += 1
    
    return right

def hillClimbing(line, points, iterations, step):
    currentSolution = line
    currentScore = quality(currentSolution, points)
    solutions = [currentSolution]

    for _ in range(iterations):
        if currentScore == 100:
            return np.array(solutions)

        c = candidate(currentSolution, step)
        cs = quality(c, points)

        if cs >= currentScore:
            currentScore = cs
            currentSolution = c
            solutions.append(currentSolution)
        
    return np.array(solutions)

solutions = hillClimbing(solution, points, 5000, 50)

plot(solutions, points)