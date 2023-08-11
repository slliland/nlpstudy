# -*- 本脚本用于绘制模型损失值变化 Created by Songyujian 2023.8.3-*-
from matplotlib import pyplot

pyplot.rcParams['font.sans-serif'] = ['SimHei']  # 设置加载的字体名
pyplot.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

pyplot.figure(figsize=(20, 10), dpi=100)
time = [1, 2, 3, 4, 5, 6, 7]
loss = [3.5935986042022705, 3.0654447078704834, 3.0060856342315674, 2.9394426345825195, 2.87406063079834,
        2.8246777057647705, 2.780829429626465]
val_loss = [2.988863229751587, 2.9535317420959473, 2.8629515171051025, 2.848977565765381, 2.754831075668335,
            2.711277961730957, 2.728508949279785]
pyplot.plot(time, loss, c='red', label=u'训练损失值')
pyplot.plot(time, val_loss, c='blue', label=u'测试损失值')
pyplot.grid(True, linestyle='--', alpha=0.5)
pyplot.xlabel("epochs", fontdict={'size': 16})
pyplot.ylabel("value", fontdict={'size': 16})
pyplot.title(u"模型随时间变化情况", fontdict={'size': 20})
pyplot.legend(loc="best")
pyplot.show()
