How to use
--------------------

Please use [docker file](../Dockerfile/Dockerfile).

And then run the file.

```
python main.py
```

Model configs
--------------------

```
Dataset ModelNet40
The point number and the nearest neighbor number is 1024 and 40
The first and second layer filter number is 1000 and 1000
The fc neuron number is 600 and the output number is 40
The Chebyshev polynomial order for each layer are 4 and 3
The weighting scheme is weighted and the weighting scaler is 40
The output of the first gcn layer is Tensor("concat:0", shape=(?, 2000), dtype=float32)
Tensor("concat:0", shape=(?, 2000), dtype=float32)
The output of the second gcn layer is Tensor("concat_1:0", shape=(?, 2000), dtype=float32)
The global feature is Tensor("dropout_2/mul:0", shape=(?, 4000), dtype=float32)
The output of the first fc layer is Tensor("dropout_3/mul:0", shape=(?, 600), dtype=float32)
The output of the second fc layer is Tensor("add_3:0", shape=(?, 40), dtype=float32)
```

Results
--------------------
