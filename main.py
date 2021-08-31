#!/usr/local/bin/python

import numpy as np
import time
from rbfn import RBFN
from lwr import LWR
from line import Line
from batch import Batch

# Linear Least Squares

batch = Batch()
model = Line(batch.batch_size)
batch.make_linear_batch_data()
start = time.process_time()
model.train(batch.x_data, batch.y_data)
print("LLS time:", time.process_time() - start)
model.plot(batch.x_data, batch.y_data)

start = time.process_time()
model.train_from_stats(batch.x_data, batch.y_data)
print("LLS from scipy stats:", time.process_time() - start)
model.plot(batch.x_data, batch.y_data)

start = time.process_time()
model.train_regularized(batch.x_data, batch.y_data, coef=0.01)
print("regularized LLS :", time.process_time() - start)
model.plot(batch.x_data, batch.y_data)

# Batch RBFNs

model = RBFN(nb_features=10)
batch.make_nonlinear_batch_data()

start = time.process_time()
model.train_ls(batch.x_data, batch.y_data)
print("RBFN LS time:", time.process_time() - start)
model.plot(batch.x_data, batch.y_data)

start = time.process_time()
model.train_ls2(batch.x_data, batch.y_data)
print("RBFN LS2 time:", time.process_time() - start)
model.plot(batch.x_data, batch.y_data)

# Incremental RBFNs

max_iter = 50
model = RBFN(nb_features=10)
start = time.process_time()

# Generate a batch of data and store it
batch.reset_batch()
for i in range(max_iter):
    x, y = batch.add_non_linear_sample()
    model.train_gd(x, y, alpha=0.5)
    # model.train_rls(x, y)
    # model.train_rls_sherman_morrison(x, y)

print("RBFN Incr time:", time.process_time() - start)
model.plot(batch.x_data, batch.y_data)

# LWR

model = LWR(nb_features=10)
batch.make_nonlinear_batch_data()

start = time.process_time()
model.train_lwls(batch.x_data, batch.y_data)
print("LWR time:", time.process_time() - start)
model.plot(batch.x_data, batch.y_data)
