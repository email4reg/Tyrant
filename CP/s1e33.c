#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int global_var1;
int global_var2;
static int file_static_var1;
static int file_static_var2;

void func1(int func1_param1, int func1_param2)
{
    static int func1_static_var1;
    static int func1_static_var2;

    // 输出行参的地址
    printf("addr of func1_param1: %010p\n", &func1_param1);
    printf("addr of func1_param2: %010p\n", &func1_param2);

    // 输出静态局部变量的地址
    printf("addr of func1_static_var1: %010p\n", &func1_static_var1);
    printf("addr of func1_static_var2: %010p\n", &func1_static_var2);
}

void func2(const int func2_const_param1, const int func2_const_param2)
{
    int func2_var1;
    int func2_var2;

    // 输出const参数的地址
    printf("addr of func2_const_param1: %010p\n", &func2_const_param1);
    printf("addr of func2_const_param2: %010p\n", &func2_const_param2);

    // 输出局部变量的地址
    printf("addr of func2_var1: %010p\n", &func2_var1);
    printf("addr of func2_var2: %010p\n", &func2_var2);
}

int main1(void)
{
    char *string1 = "I love FishC.com";
    char *string2 = "very much";

    // 输出函数的地址
    printf("addr of func1: %010p\n", func1);
    printf("addr of func2: %010p\n", func2);

    // 输出字符串常量的地址
    printf("addr of string1: %010p\n", string1);
    printf("addr of string2: %010p\n", string2);

    // 输出全局变量的地址
    printf("addr of global_var1: %010p\n", &global_var1);
    printf("addr of global_var2: %010p\n", &global_var2);

    // 输出文件内的static变量的地址
    printf("addr of file_static_var1: %010p\n", &file_static_var1);
    printf("addr of file_static_var2: %010p\n", &file_static_var2);

    // 输出函数内局部变量的地址
    func1(1, 2);
    func2(3, 4);

    return 0;
}

void createMatrix(int n);

void createMatrix(int n)
{
    int i, j, oi, oj, num, max;

    // 创建一个存放矩阵的二维数组
    int matrix[n][n];

    // 填充为0
    max = n * n;
    memset(matrix, 0, max * sizeof(int));

    // 将1存放到第0行的中间位置
    matrix[0][n / 2] = 1;

    // 记录当前的行号和列号
    i = 0;
    j = n / 2;

    for (num = 2; num <= max; num++)
    {
        // 记住当前位置
        oi = i;
        oj = j;

        // 向右上角走一步
        i = i - 1;
        j = j + 1;

        if (i < 0)
        {
            i = n - 1;
        }

        if (j > n - 1)
        {
            j = 0;
        }

        if (matrix[i][j] != 0)
        {
            i = oi + 1;
            j = oj;
        }

        matrix[i][j] = num;
    }

    // 打印结果
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("%2d   ", matrix[i][j]);
        }
        putchar('\n');
    }
}

int main(void)
{
    int n;

    printf("请输入一个奇数：");
    scanf("%d", &n);
    while (!(n % 2) || n < 3)
    {
        printf("输入错误，必须是一个大于2的奇数！\n");
        printf("请重新输入：");
        scanf("%d", &n);
    }

    createMatrix(n);

    return 0;
}