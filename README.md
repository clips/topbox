# shed
A small Python 3 wrapper around the Stanford Topic Modeling Toolbox (STMT) that makes working with L-LDA a bit easier; no need to leave the Python environment. More information on its workings can be found on [my blog](https://cmry.github.io/2015/06/18/shed/).

# Example

``` python
import shed

stmt = shed.STMT('bit_of_testing', epochs=10, mem=15000)


space = ['text text more text', 'things to do with text']
labels = ['label1 label2', 'label1 label3']

stmt.train(space, labels)


infer = ['this is a text', 'things with more text']
gs = ['label1 label2', 'label1 label3']

stmt.test(infer, gs)


from sklearn.metrics import average_precision_score

# array requires numpy and scipy
y_true, y_score = stmt.results(gs, array=True)

print(average_precision_score(y_true, y_score))
```