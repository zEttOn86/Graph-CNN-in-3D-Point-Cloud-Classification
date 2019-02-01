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
Total parameters number is 5438640
```

Results
--------------------
```
===========================epoch 0====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 4.35256433487 0.0391351357102 and 0.362516015768======
the average acc among 40 class is:0.400854651163
===========average loss and acc for this epoch is 1.78842818737 and 0.506888151169=======
===========================epoch 1====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 2.8364379406 0.0392088294029 and 0.561001121998======
the average acc among 40 class is:0.558610465116
===========average loss and acc for this epoch is 1.27902507782 and 0.63897895813=======
===========================epoch 2====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 2.22493863106 0.0404869504273 and 0.648021101952======
the average acc among 40 class is:0.608029069767
===========average loss and acc for this epoch is 0.997375786304 and 0.700162053108=======
===========================epoch 3====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 1.90396821499 0.0419263355434 and 0.69669675827======
the average acc among 40 class is:0.680156976744
===========average loss and acc for this epoch is 0.841318368912 and 0.746758520603=======
===========================epoch 4====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 1.68683755398 0.0434363000095 and 0.72694170475======
the average acc among 40 class is:0.709755813953
===========average loss and acc for this epoch is 0.758493959904 and 0.769448935986=======
===========================epoch 5====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 1.51005959511 0.0448392443359 and 0.75374519825======
the average acc among 40 class is:0.707965116279
===========average loss and acc for this epoch is 0.696989655495 and 0.784035682678=======
===========================epoch 6====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 1.39878070354 0.046168576926 and 0.772363126278======
the average acc among 40 class is:0.755459302326
===========average loss and acc for this epoch is 0.637444615364 and 0.806726098061=======
===========================epoch 7====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 1.31010329723 0.0475177355111 and 0.784303963184======
the average acc among 40 class is:0.746220930233
===========average loss and acc for this epoch is 0.64052426815 and 0.801053464413=======
===========================epoch 8====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 1.22422230244 0.048816613853 and 0.802881419659======
the average acc among 40 class is:0.770941860465
===========average loss and acc for this epoch is 0.582386672497 and 0.8237439394=======
===========================epoch 9====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 1.18051338196 0.0500744171441 and 0.801874339581======
the average acc among 40 class is:0.771796511628
===========average loss and acc for this epoch is 0.568809151649 and 0.817666113377=======
===========================epoch 10====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 1.13195371628 0.0512037798762 and 0.812082409859======
the average acc among 40 class is:0.785552325581
===========average loss and acc for this epoch is 0.527403175831 and 0.831442475319=======
===========================epoch 11====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 1.10256838799 0.0523296780884 and 0.819677472115======
the average acc among 40 class is:0.784151162791
===========average loss and acc for this epoch is 0.536724090576 and 0.820097267628=======
===========================epoch 12====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 1.05340480804 0.0534010604024 and 0.827522099018======
the average acc among 40 class is:0.801058139535
===========average loss and acc for this epoch is 0.489861398935 and 0.854132890701=======
===========================epoch 13====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 1.01384532452 0.0543872341514 and 0.831751525402======
the average acc among 40 class is:0.776860465116
===========average loss and acc for this epoch is 0.530577003956 and 0.826985418797=======
===========================epoch 14====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 0.990660667419 0.0555504076183 and 0.836218714714======
the average acc among 40 class is:0.80563372093
===========average loss and acc for this epoch is 0.4977876544 and 0.843192875385=======
===========================epoch 15====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 0.965846717358 0.0565502867103 and 0.836603343487======
the average acc among 40 class is:0.807598837209
===========average loss and acc for this epoch is 0.476660132408 and 0.855348467827=======
===========================epoch 16====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 0.946036815643 0.0575121715665 and 0.840071856976======
the average acc among 40 class is:0.811970930233
===========average loss and acc for this epoch is 0.474046707153 and 0.854943275452=======
===========================epoch 17====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 0.912372231483 0.0585295073688 and 0.847071349621======
the average acc among 40 class is:0.803470930233
===========average loss and acc for this epoch is 0.470172405243 and 0.854943275452=======
===========================epoch 18====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 0.88787996769 0.0594591684639 and 0.852905094624======
the average acc among 40 class is:0.815046511628
===========average loss and acc for this epoch is 0.46793705225 and 0.853322505951=======
===========================epoch 19====================
0.0006
=============average loss, l2 loss, acc  for this epoch is 0.869523227215 0.0604129508138 and 0.855044186115======
the average acc among 40 class is:0.809546511628
===========average loss and acc for this epoch is 0.46435931325 and 0.85210698843=======
===========================epoch 20====================
0.0003
=============average loss, l2 loss, acc  for this epoch is 0.792204916477 0.0609394200146 and 0.872469425201======
the average acc among 40 class is:0.833098837209
===========average loss and acc for this epoch is 0.408683717251 and 0.872366309166=======
===========================epoch 21====================
0.0003
=============average loss, l2 loss, acc  for this epoch is 0.746579945087 0.060851406306 and 0.877149283886======
the average acc among 40 class is:0.833011627907
===========average loss and acc for this epoch is 0.415133446455 and 0.873176634312=======
===========================epoch 22====================
0.0003
=============average loss, l2 loss, acc  for this epoch is 0.729416489601 0.0608044490218 and 0.877822399139======
the average acc among 40 class is:0.833627906977
===========average loss and acc for this epoch is 0.43379753828 and 0.869124770164=======
===========================epoch 23====================
0.0003
=============average loss, l2 loss, acc  for this epoch is 0.715777397156 0.0607511810958 and 0.882058501244======
the average acc among 40 class is:0.838930232558
===========average loss and acc for this epoch is 0.413148075342 and 0.873987019062=======
===========================epoch 24====================
0.0003
=============average loss, l2 loss, acc  for this epoch is 0.695435285568 0.0607278570533 and 0.885896503925======
the average acc among 40 class is:0.843296511628
===========average loss and acc for this epoch is 0.401672244072 and 0.879254460335=======
===========================epoch 25====================
0.0003
=============average loss, l2 loss, acc  for this epoch is 0.686274409294 0.0607049539685 and 0.884801566601======
the average acc among 40 class is:0.842546511628
===========average loss and acc for this epoch is 0.415737658739 and 0.869935154915=======
===========================epoch 26====================
0.0003
=============average loss, l2 loss, acc  for this epoch is 0.676807105541 0.0606955699623 and 0.885823905468======
the average acc among 40 class is:0.824976744186
===========average loss and acc for this epoch is 0.435424417257 and 0.864667773247=======
===========================epoch 27====================
0.0003
=============average loss, l2 loss, acc  for this epoch is 0.67456638813 0.0607220120728 and 0.888543426991======
the average acc among 40 class is:0.840970930233
===========average loss and acc for this epoch is 0.416981160641 and 0.876418173313=======
===========================epoch 28====================
0.0003
=============average loss, l2 loss, acc  for this epoch is 0.655502676964 0.0607698187232 and 0.889508426189======
the average acc among 40 class is:0.831348837209
===========average loss and acc for this epoch is 0.427979409695 and 0.865883290768=======
===========================epoch 29====================
0.0003
=============average loss, l2 loss, acc  for this epoch is 0.66287124157 0.0608703568578 and 0.889725983143======
the average acc among 40 class is:0.838011627907
===========average loss and acc for this epoch is 0.41294926405 and 0.876012980938=======
===========================epoch 30====================
0.0003
=============average loss, l2 loss, acc  for this epoch is 0.656663119793 0.0609544105828 and 0.890635311604======

```
