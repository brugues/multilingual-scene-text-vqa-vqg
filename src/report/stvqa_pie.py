import json
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filename = '../data/ST-VQA/annotations/stvqa_train.json'

    with open(filename, 'r') as f:
        stvqa = json.load(f)

    answer_words = {
        1: 0,
        5: 0,
        3: 0,
        4: 0,
        2: 0
    }

    for instance in stvqa:
        if len(instance['answer']) >= 5:
            answer_words[5] += 1
        else:
            answer_words[len(instance['answer'])] += 1

    # Pie Plot
    keys = answer_words.keys()
    labels = []
    for label in keys:
        if label == 1:
            labels.append('{} word answer\n({}, {:0.2f}%)'.format(label, answer_words[label],
                                                           answer_words[label]/len(stvqa)*100))
        elif label == 5:
            labels.append('{}+ words answer\n({}, {:0.2f}%)'.format(label, answer_words[label],
                                                             answer_words[label]/len(stvqa)*100))
        else:
            labels.append('{} words answer\n({}, {:0.2f}%)'.format(label, answer_words[label],
                                                            answer_words[label]/len(stvqa)*100))

    fig, ax = plt.subplots(figsize=(4.75, 3.15), subplot_kw=dict(aspect="equal"))
    plt.pie(answer_words.values(), labels=labels, shadow=True, startangle=90,
            labeldistance=1.2)
    plt.show()
    fig.savefig("pie_stvqa.png")

    # fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    #
    # data = list(answer_words.values())
    #
    # wedges, texts = ax.pie(data, wedgeprops=dict(width=0.75), startangle=85, shadow=True)
    #
    # kw = dict(arrowprops=dict(arrowstyle="-"), zorder=0, va="center")
    #
    # for i, p in enumerate(wedges):
    #     ang = (p.theta2 - p.theta1) / 2. + p.theta1
    #     y = np.sin(np.deg2rad(ang))
    #     x = np.cos(np.deg2rad(ang))
    #     horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    #     connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    #     kw["arrowprops"].update({"connectionstyle": connectionstyle})
    #     ax.annotate(labels[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
    #                 horizontalalignment=horizontalalignment, **kw)
    #
    # ax.legend(wedges, labels,
    #           title="Number of words per answer",
    #           loc="center left",
    #           bbox_to_anchor=(-0.85, 0.25, 0, 1.22))
    #
    # plt.show()
