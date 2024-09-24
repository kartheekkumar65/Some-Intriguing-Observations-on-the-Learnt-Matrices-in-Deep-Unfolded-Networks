# Expandability of deep unfolded networks  

## Setting Up a Virtual Environment

### Install virtualenv

install `virtualenv` if it's not already installed:
```bash
pip install virtualenv
```

### Create a Virtual Environment

```bash
virtualenv myenv
```

### Activate the Virtual Environment

#### macOS and Linux:

```bash
myenv\Scripts\activate
```

#### Windows:

```bash
source myenv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Run

The project's main functionality is centered around the `./src/main.py` script. Here are the steps to execute different configurations of the algorithm:

### Standard Execution

To run the DUN's unbiased algorithm, modify the `./src/main.py` file:

- Comment or uncomment line number 95, 96, or 97 as needed to select the desired algorithm variant. This setup will train the DUN and display the results.

### Custom Parameters

You can customize the execution parameters by passing values through command-line arguments.

### Running the Iterative Algorithm

To execute the iterative counterpart of an algorithm, you must provide specific arguments for the number of iterations and threshold value:

#### Example Command

To run the iterative algorithm, use the following command:
```bash
python ./src/main.py --num_iter 500 --thr_val 0.06
```
Adjust the `--num_iter` and `--thr_val` arguments to meet your specific requirements.

#### How to choose the threshold value

Typically, the thr_val ranges from $0.001$ to $0.1$. Lower noise levels benefit from smaller threshold values, such as $0.006$. For higher noise levels, a higher threshold value is preferred, generally greater than $0.01$. Adjust the threshold value according to the noise level for optimal performance of these iterative methods.
