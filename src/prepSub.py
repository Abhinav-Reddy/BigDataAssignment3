import pandas as pd
import numpy as np
import re
import seaborn as sns

import itertools
import datetime

import re
import urllib
import os

import numpy as np

test_df = pd.read_csv("../results/sub.csv")
num = []
for index, row in test_df.iterrows():
        if row['prediction'] > 0.4:
                num.append(1)
        else:
                num.append(0)

sub_df = pd.DataFrame(data=num,columns={"prediction"}, dtype=int)
sub_df.to_csv(path_or_buf="../results/Finalsub.csv", columns={"prediction"}, header=True, index=True, index_label="id")




