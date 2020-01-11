#include <stdio.h>
#include <string.h>
#include <math.h>

#define MAX 1024

int func()
{
    char *p[4] = {"hello","world","iam","haoran"};
    char *(*pa)[4] = &p;

    printf("*(*pa) = %s\n", *(*pa));
    printf("**pa = %s\n", **pa);
    printf("***pa = %c\n", ***pa);
    printf("***pa + 1 = %c\n", ***pa + 1);
    printf("*(*pa + 1) = %s\n", *(*pa + 1));
    printf("\n");

    printf("**pa = %p\n", **pa);
    printf("p[0] = %p\n", p[0]);
    printf("\n");

    printf("*pa = %p\n",*pa);
    printf("&p = %p\n", &p);
    printf("pa = %p\n", pa);
    printf("\n");

    printf("pa + 1 = %p\n", pa + 1);
    printf("*(pa + 1) = %p\n", *(pa + 1));
    printf("*pa + 1 = %p\n", *pa + 1);
    printf("\n");

    return 0;
}

int func1()
// note: 指针p指向数组中的某个元素的地址,那么p+1就表示下一个元素的地址
{
    int array[3][4] = {1,2,3,4,5,6,7,8,9,10,11,12};
    int (*p)[3][4] = &array;
    /* p指向一个二维数组,*p则表示取出这个二维数组array, (*p + 1)与(array + 1)是等价的
    表示取出这个数组的第二个元素(在二维数组中的元素是一个一维数组),指向第一个一维数组的第一个元素的首地址 */
    printf("*(*p + 1) = %p\n", *(*p + 1));
    printf("&array[1] = %p\n", &array[1]);

    return 0;
}

void func2()
{
    int matrix[3][3];
    int (*p)[3][3] = &matrix;
    int i,j;

    printf("请输入一个字符串:");
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            matrix[i][j] = getchar();
        }
    }

    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            printf("%c ", matrix[i][j]);
        }
        printf("\n");
    }

    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            printf("%c ", (*p)[i][j]); // 若*p[i][j]，则根据优先级就是先p[i][j]然后*,而p仅是一个指针.这里的()仅仅是为提高优先级,与定义不同
        }
        printf("\n");
    }

    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            printf("%c ", *(*(*p + i) + j)); // *p == matrix. *(*(*p + i) + j)是(*p)[i][j]的指针形式
        }
        printf("\n");
    }
}

void func3()
{
    int length, aver;
    int i, j;
    char str[MAX];

    scanf("%s", str);

    length = strlen(str);
    aver = sqrt(length);

    for (i = 0; i < aver; i++)
    {
        for (j = 0; j < aver; j++)
        {
            printf("%c ", str[i * aver + j]);
        }
        printf("\n");
    }
}

void func4()
{
    float array[] = {
        31.3, 35.5, 58.7, 49.6, 55.5,
        59.8, 54.9, 33.1, 38.2, 26.6, 20.5, 27.8, 38.5, 41.5, 44.7, 38.1, 41.5,
        34.9, 36.4, 47.5, 37.9, 30.6, 23.4, 26.6, 34.3};
    int year, month, index;
    
    printf("请输入待查询年月份(年-月): ");
    scanf("%d-%d", &year, &month);

    index = (year - 2014) * 12 + month;
    if (index < 8 || index > 32)
    {
        printf("抱歉, 未收录该日期数据!\n");
    }
    else
    {
        printf("%d年%d月广州的PM2.5值是: %.2f\n", year, month, array[index - 8]);
    }
}

int func5()
{
    float pm25[3][12] = {
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 31.3, 35.5, 58.7, 49.6, 55.5},
        {59.8, 54.9, 33.1, 38.2, 26.6, 20.5, 27.8, 38.5, 41.5, 44.7, 38.1, 41.5},
        {34.9, 36.4, 47.5, 37.9, 30.6, 23.4, 26.6, 34.3, 0.0, 0.0, 0.0, 0.0}};
    int i, j, step;
    float min, max, data;

    // 找出最大值和最小值
    min = max = pm25[1][0];
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 12; j++)
        {
            if (pm25[i][j])
            {
                min = min > pm25[i][j] ? pm25[i][j] : min;
                max = max < pm25[i][j] ? pm25[i][j] : max;
            }
        }
    }

    // 计算步进值
    if ((int)(max - min) > 80)
    {
        step = 2;
    }
    else
    {
        step = 1;
    }

    printf("最小值: %.2f, 最大值: %.2f\n", min, max);
    // 打印直方图
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 12; j++)
        {
            data = pm25[i][j];
            if (data)
            {
                printf("%d年%2d月: ", i + 2014, j + 1);
                while (data >= min)
                {
                    printf("*");
                    data -= step;
                }
                printf("\n");
            }
        }
    }

    return 0;
}

int main()
{
    // func();
    // func1();
    // func2();
    // func3();
    // func4();
    func5();

    return 0;
}
