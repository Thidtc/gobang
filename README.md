# Gobang
A PyTorch-based implementation of AlphaGo Zero algorithm for gobang game.
Also, provide a OpenGym style environment interface for gobang game

# Run

## training

``` python
python train.py
```

# Result
## 5\*5 board, n_in_row is 4
* training time

  2~3 hours

* results

  9:1 versus pure monte carlo tree search algorithm(with 10 times sample counts)

# TODO
* Add interface for human 
* Implement render_ interface for gobang environment
