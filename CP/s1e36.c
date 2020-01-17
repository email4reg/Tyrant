#include <stdio.h>

int count = 0;

int check(int i, int j, int (*queen)[4]);
void setQueen(int i, int (*queen)[4]);

int check(int i, int j, int (*queen)[4])
{
    int s, t;

    // 判断行
    for (s = i, t = 0; t < 4; t++)
    {
        if (queen[s][t] == 1 && t != j)
        {
            return 0;
        }
    }

    // 判断列
    for (t = j, s = 0; s < 4; s++)
    {
        if (queen[s][t] == 1 && s != i)
        {
            return 0;
        }
    }

    // 判断左上方
    for (s = i - 1, t = j - 1; s >= 0 && t >= 0; s--, t--)
    {
        if (queen[s][t] == 1)
        {
            return 0;
        }
    }

    // 判断右上方
    for (s = i + 1, t = j + 1; s < 4 && t < 4; s++, t++)
    {
        if (queen[s][t] == 1)
        {
            return 0;
        }
    }

    // 经过上面层层关卡还能存活，那么说明符合条件,返回1
    return 1;
}

void setQueen(int col, int (*queen)[4])
{
    int i, j, row;

    // 所有皇后放置完毕
    if (col == 4)
    {
        for (i = 0; i < 4; i++)
        {
            for (j = 0; j < 4; j++)
            {
                if (queen[i][j] != 0)
                {
                    printf("Q ");
                }
                else
                {
                    printf("* ");
                }
            }
            putchar('\n');
        }

        putchar('\n');
        count++;

        return;
    }

    // 迭代每一行
    for (row = 0; row < 4; row++)
    {
        // 检查每一行中对应的每一列能否放置皇后
        if (check(row, col, queen))
        {
            // 如果queen[row][col]符合条件，则放置皇后
            queen[row][col] = 1;
            // col+1，进入下一层递归
            setQueen(col + 1, queen);
            // 只有两种情况会执行下面语句
            // 1. col+1遇到所有的row都不合适
            // 2. 完成整个二维数组的放置
            // 无论哪种情况，
            queen[row][col] = 0;
        }
    }
}

int main1(void)
{
    int queen[4][4];
    int i, j;

    // 初始化二维数组，1表示已放置皇后，0表示没有
    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 4; j++)
        {
            queen[i][j] = 0;
        }
    }

    setQueen(0, queen);

    return 0;
}

#define X 8
#define Y 8

int chess[X][Y];

// 找到下一个可走的位置
int next(int *px, int *py, int count)
{
    int x = *px;
    int y = *py;

    switch (count)
    {
    case 0:
        if (x + 2 <= X - 1 && y - 1 >= 0 && chess[x + 2][y - 1] == 0)
        {
            *px = x + 2;
            *py = y - 1;
            return 1;
        }
        break;
    case 1:
        if (x + 2 <= X - 1 && y + 1 <= Y - 1 && chess[x + 2][y + 1] == 0)
        {
            *px = x + 2;
            *py = y + 1;
            return 1;
        }
        break;
    case 2:
        if (x + 1 <= X - 1 && y - 2 >= 0 && chess[x + 1][y - 2] == 0)
        {
            *px = x + 1;
            *py = y - 2;
            return 1;
        }
        break;
    case 3:
        if (x + 1 <= X - 1 && y + 2 <= Y - 1 && chess[x + 1][y + 2] == 0)
        {
            *px = x + 1;
            *py = y + 2;
            return 1;
        }
        break;
    case 4:
        if (x - 2 >= 0 && y - 1 >= 0 && chess[x - 2][y - 1] == 0)
        {
            *px = x - 2;
            *py = y - 1;
            return 1;
        }
        break;
    case 5:
        if (x - 2 >= 0 && y + 1 <= Y - 1 && chess[x - 2][y + 1] == 0)
        {
            *px = x - 2;
            *py = y + 1;
            return 1;
        }
        break;
    case 6:
        if (x - 1 >= 0 && y - 2 >= 0 && chess[x - 1][y - 2] == 0)
        {
            *px = x - 1;
            *py = y - 2;
            return 1;
        }
        break;
    case 7:
        if (x - 1 >= 0 && y + 2 <= Y - 1 && chess[x - 1][y + 2] == 0)
        {
            *px = x - 1;
            *py = y + 2;
            return 1;
        }
        break;
    }

    return 0;
}

int setHorse(int x, int y, int tag)
{
    int x1 = x, y1 = y, flag = 0, count = 0;

    // tag记录轨迹
    chess[x][y] = tag;
    // 如果tag等于64退出程序
    if (tag == X * Y)
    {
        return 1;
    }

    // 如果可以走，那么flag为1
    flag = next(&x1, &y1, count);
    // 否则尝试其他路径
    while (flag == 0 && count < 7)
    {
        count += 1;
        flag = next(&x1, &y1, count);
    }

    // 递归进入下一个坐标
    while (flag)
    {
        // 返回1表示成功找到落脚点
        if (setHorse(x1, y1, tag + 1))
        {
            return 1;
        }
        // 否则从上一步重新尝试
        x1 = x;
        y1 = y;
        count += 1;
        flag = next(&x1, &y1, count);
        while (flag == 0 && count < 7)
        {
            count += 1;
            flag = next(&x1, &y1, count);
        }
    }

    if (flag == 0)
    {
        chess[x][y] = 0;
    }

    return 0;
}

int main(void)
{
    int i, j;

    for (i = 0; i < X; i++)
    {
        for (j = 0; j < Y; j++)
        {
            chess[i][j] = 0;
        }
    }

    // 讲道理，从 (2, 0) 坐标开始计算是比较容易出结果的
    // 如果你比较有耐心，或 CPU 特别强劲，可以尝试计算其它坐标
    if (setHorse(2, 0, 1))
    {
        for (i = 0; i < X; i++)
        {
            for (j = 0; j < Y; j++)
            {
                printf("%02d  ", chess[i][j]);
            }
            putchar('\n');
        }
    }
    else
    {
        printf("可惜无解！\n");
    }

    return 0;
}