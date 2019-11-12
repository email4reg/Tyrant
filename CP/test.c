#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define N 30 // 最多的学生数
#define M 10 // 最多的课程数

int func1()
{
    double height = 100;
    double sum = 0;
    int i;

    for (i = 1; i <= 10; i++)
    {
        sum += 1.5 * height;
        height = height * 0.5;
        printf("落地第%d次,反弹高度为:%.5f(m),总共经过:%.5f(m)\n", i, height, sum);
    }

    return 0;
}

void input(float a[N][M], int n, int m)
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        printf("请输入%d号学生%d门课的成绩: ", i + 1, m);

        for (j = 0; j < m; j++)
        {
            scanf("%f", &a[i][j]);
        }
    }
}

void stu_mean(float a[N][M], int n, int m)
{
    int i, j;
    float sum;

    for (i = 0; i < n; i++)
    {
        sum = 0;
        for (j = 0; j < m; j++)
        {
            sum += a[i][j];
        }
        printf("%d号学生的成绩为:", i+1);

        for (j = 0; j < m; j++)
        {
            printf("%6.2f", a[i][j]);
        }
        printf("平均分为: %6.2f\n", sum/m);
    }
}

void cour_mean(float a[N][M], int n, int m)
{
    int i, j;
    float sum;

    for (i = 0; i < m; i++)
    {
        sum = 0;

        for (j = 0; j < n; j++)
        {
            sum += a[j][i];
        }
        printf("第%d门课的平均成绩为: %6.2f\n", i+1, sum/n);
    }
}

int main()
{
    int m, n;
    float a[N][M];

    printf("输入学生数n, 和课程数m: ");
    scanf("%d %d", &n, &m);

    input(a, n, m);
    stu_mean(a, n, m);
    cour_mean(a, n, m);

    return 0;

}
