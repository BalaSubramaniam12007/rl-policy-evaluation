## AIM
To simulate the Frozen-lake MDP and compare different policy functions.

## PROBLEM STATEMENT
The problem involves simulating a Frozen-lake MDP and defining various policy functions for it, these policy functions are later evaluated by a policy_evaluation() function which compares the value function of the policies passed as parameter. This is an experiment in reinforcement learning where you test different policies in FrozenLake, both by simulation (probability of reaching the goal) and by formal policy evaluation (computing expected long-term rewards).

## POLICY EVALUATION FUNCTION

<img width="685" height="130" alt="image" src="https://github.com/user-attachments/assets/834db01d-47b9-40d8-895e-5b7fc488ed1d" />

```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state] * (not done))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V
```

## OUTPUT:

### policies:
<img width="726" height="485" alt="image" src="https://github.com/user-attachments/assets/a3d23bf6-43d8-420a-aed5-33a7c9245cf4" />
<img width="824" height="109" alt="image" src="https://github.com/user-attachments/assets/b915ccd9-f4ac-465d-8f7a-a3e2405a6198" />

<img width="699" height="644" alt="image" src="https://github.com/user-attachments/assets/4957004d-854e-45ea-8c76-217dc8b86c70" />
<img width="821" height="108" alt="image" src="https://github.com/user-attachments/assets/e1834754-ae37-4325-a584-7c9f128eca9d" />



### State value function:

<img width="1015" height="286" alt="image" src="https://github.com/user-attachments/assets/2cfb2d1f-4474-4c9e-92f3-0ee346c7fe6a" />

### Compare:
<img width="581" height="178" alt="image" src="https://github.com/user-attachments/assets/b03ccb02-b8cf-4664-b927-c1ce9ea284f2" />

<img width="525" height="171" alt="image" src="https://github.com/user-attachments/assets/f8450f00-27a1-47bf-bf1d-6955ceabf66a" />

### Best Policy:
<img width="580" height="170" alt="image" src="https://github.com/user-attachments/assets/5f759492-7f80-4218-b819-51293f435ae1" />

## RESULT:
Thus we have successfully evaluated two different policies for a given env and compared their values functions.
