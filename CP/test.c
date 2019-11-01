#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


int main()
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
