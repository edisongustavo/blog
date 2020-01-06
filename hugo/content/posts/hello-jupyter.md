+++
title = "Publishing with Jupyter notebooks"
date = "2020-01-06"
categories = ["Jupyter", "Hugo"]
subtitle = "How to publish to Hugo using jupyter notebooks"
tags = ["Jupyter", "Notebook", "Hugo", "nb2hugo"]
+++


My intention here is to show that it is indeed possible to use Jupyter to write blog posts, which I actually intend to do more often than writing Markdown

# Why?

Jupyter is great because I can mix code and documentation together.


```python
print("Text is supported :)")
```

    Text is supported :)


For example, look at this example from the [seaborn](https://seaborn.pydata.org/) library:


```python
!pip install seaborn
```


```python
import seaborn as sns
sns.set()
tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", col="time",
            hue="smoker", style="smoker", size="size",
            data=tips);
```


![png](output_6_0.png)

