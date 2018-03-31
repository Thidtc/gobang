# Gobang
This is a PyTorch-based implementation of AlphaGo Zero algorithm for gobang game.
Also, provide an OpenAI Gym style environment interface for gobang game

## Environment
Python 3.*
PyTorch 0.3.*

## Usage

Training

``` bash
python train.py
```

## Result
### 6\*6 board, n_in_row is 4
* training time

  2~3 hours

* results

  9:1 versus pure monte carlo tree search algorithm(with 10 times sample counts)

## TODO
* Add interface for human player
* Implement render_ interface for gobang environment so as to provide an UI
