{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Faktoranalyse\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import sklearn\n",
    "from sklearn.decomposition import FactorAnalysis\n",
    "from sklearn import datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[5.1, 3.5, 1.4, 0.2],\n",
       "       [4.9, 3. , 1.4, 0.2],\n",
       "       [4.7, 3.2, 1.3, 0.2],\n",
       "       [4.6, 3.1, 1.5, 0.2],\n",
       "       [5. , 3.6, 1.4, 0.2],\n",
       "       [5.4, 3.9, 1.7, 0.4],\n",
       "       [4.6, 3.4, 1.4, 0.3],\n",
       "       [5. , 3.4, 1.5, 0.2],\n",
       "       [4.4, 2.9, 1.4, 0.2],\n",
       "       [4.9, 3.1, 1.5, 0.1]])"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iris =  datasets.load_iris()\n",
    "X = iris.data\n",
    "variable_names = iris.feature_names\n",
    "X[0:10,]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "Faktoranalyse auf dem iris-Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>0.706989</td>\n",
       "      <td>-0.158005</td>\n",
       "      <td>1.654236</td>\n",
       "      <td>0.70085</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>0.115161</td>\n",
       "      <td>0.159635</td>\n",
       "      <td>-0.044321</td>\n",
       "      <td>-0.01403</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-0.00000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)\n",
       "0           0.706989         -0.158005           1.654236           0.70085\n",
       "1           0.115161          0.159635          -0.044321          -0.01403\n",
       "2          -0.000000          0.000000           0.000000           0.00000\n",
       "3          -0.000000          0.000000           0.000000          -0.00000"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "factor = FactorAnalysis().fit(X)\n",
    "pd.DataFrame(factor.components_, columns=variable_names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
