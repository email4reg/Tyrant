#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define M 2
#define N 2
#define P 3

#define NUM 5

int func1()
{
    int a[4][5] = 
    {
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20}
    };

    int i, j;

    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 5; j++)
        {
            printf("%2d ", a[i][j]);
            if (i + j == 3)
            {
                printf("\n");
            }
        }
    }

    printf("\n");

    return 0;
}

int func2()
{
    int a[M][P] = {
        {1, 2, 3},
        {4, 5, 6}};

    int b[P][N] = {
        {1, 4},
        {2, 5},
        {3, 6}};

    int c[M][N] = {0};

    int i, j, k, row;

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (k = 0; k < P; k++)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    // row 取行数最大值
    row = M > P ? M : P;

    for (i = 0; i < row; i++)
    {
        // 打印A
        printf("|  ");
        for (j = 0; j < P; j++)
        {
            if (i < M)
            {
                printf("\b%d ", a[i][j]);
                printf("|");
            }
            else
            {
                printf("\b\b\b     ");
            }
        }
        // 打印 * 号
        if (i == row / 2)
        {
            printf(" * ");
        }
        else
        {
            printf("   ");
        }
        printf("|  ");
        // 打印B
        for (j = 0; j < N; j++)
        {
            if (i < P)
            {
                printf("\b%d ", b[i][j]);
                printf("|");
            }
            else
            {
                printf("\b\b\b     ");
            }
        }
        // 打印 = 号
        if (i == row / 2)
        {
            printf(" = ");
        }
        else
        {
            printf("   ");
        }
        // 打印C
        printf("|  ");
        for (j = 0; j < N; j++)
        {
            if (i < M)
            {
                printf("\b%d ", c[i][j]);
                printf("|");
            }
            else
            {
                printf("\b\b\b      ");
            }
        }
        printf("\n");
    }

    return 0;
}

int func3()
{
    char slogan[NUM][100];
    int i, j, ch, min, max, temp;

    for (i = 0; i < NUM; i++)
    {
        printf("请输入%d句话：", i + 1);
        for (j = 0; (ch = getchar()) != '\n'; j++)
        {
            slogan[i][j] = ch;
        }
        slogan[i][j] = '\0';
    }

    min = 0;
    max = min;

    printf("你输入了下边%d句话：\n", NUM);

    // 打印每句口号，同时比较长度
    for (i = 0; i < NUM; i++)
    {
        printf("%s\n", slogan[i]);
        temp = strlen(slogan[i]);
        min = temp < strlen(slogan[min]) ? i : min;
        max = temp > strlen(slogan[max]) ? i : max;
    }

    printf("其中最长的是：%s\n", slogan[max]);
    printf("其中最短的是：%s\n", slogan[min]);

    return 0;
}

int main()
{
    // func1();
    // func2();
    func3();

    return 0;
}