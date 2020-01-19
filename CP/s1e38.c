#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HUMANWIN 0
#define COMPUTERWIN 1

#define N 10

int main1(void)
{
    int *ptr = NULL;

    ptr = (int *)malloc(N * sizeof(int));
    if (ptr == NULL)
    {
        exit(1);
    }

    memset(ptr, 0, N * sizeof(int));
    for (int i = 0; i < N; i++)
    {
        printf("%d ", *(ptr + i)); // ptr[i]
    }
    putchar('\n');

    free(ptr);

    return 0;
}

int main2(void)
{
    int *ptr1 = NULL;
    int *ptr2 = NULL;

    // 第一次申请内存空间, 这里是堆空间
    ptr1 = (int *)malloc(10 * sizeof(int));

    //发生ptr1不够用, 申请ptr2
    ptr2 = (int *)malloc(20 * sizeof(int));
    // 拷贝ptr1中的数据到ptr2
    memcpy(ptr2, ptr1, 10);
    // 释放 ptr1
    free(ptr1);

    // 以上过程可以用realloc()实现，realloc(NULL, 20) == malloc(20); realloc(ptr1,0) == free(ptr1) 且ptr1 != NULL
    ptr1 = realloc(ptr1, 20);

    return 0;
}

int main3(void)
{
    int i, num;
    int count = 0;
    int *ptr = NULL; // 这里必须指向NULL

    do
    {
        printf("请输入一个正整数(输入-1结束): ");
        scanf("%d", &num);
        count++;

        ptr = (int *)realloc(ptr, count * sizeof(int));
        *(ptr + count - 1) = num;
        // ptr[count - 1] = num;

    } while (num != -1);
    
    printf("用户输入的整数分别是: ");
    for (i = 0; i < count; i++)
    {
        printf("%d ", *(ptr + i));
        // printf("%d ", ptr[i]);
    }
    putchar('\n');

    free(ptr);

    return 0;
}

int get_computer(void)
{
    int computer;

    srand((unsigned)time(NULL));
    computer = rand() % 3 + 1;

    return computer;
}

int get_human(void)
{
    int human;

    printf("请出拳（1剪刀/2石头/3布/0退出）-> ");
    scanf("%d", &human);

    while (human < 0 || human > 3)
    {
        printf("出拳错误，请重新出拳（只需要输入数字即可）-> ");
        scanf("%d", &human);
    }

    return human * 3;
}

void welcome(void)
{
    printf("\n########################\n");
    printf("# 欢迎来到猜拳小游戏！ #\n");
    printf("########################\n\n");
}

void gameover(int winner)
{
    if (winner)
    {
        printf("\n#########################################################################\n");
        printf("#                                                                       #\n");
        printf("# ##    ##    ####    ##     ##    ##         ####     ######  ######## #\n");
        printf("#  ##  ##    ##  ##   ##     ##    ##        ##  ##   ##       ##       #\n");
        printf("#   ####    ##    ##  ##     ##    ##       ##    ##  ##       ##       #\n");
        printf("#    ##     ##    ##  ##     ##    ##       ##    ##  ######   #######  #\n");
        printf("#    ##     ##    ##  ##     ##    ##       ##    ##       ##  ##       #\n");
        printf("#    ##      ##  ##    ##   ##     ##        ##  ##        ##  ##       #\n");
        printf("#    ##       ####      #####      #######    ####    ######   ######## #\n");
        printf("#                                                                       #\n");
        printf("#########################################################################\n");
    }
    else
    {
        printf("\n##########################################################################\n");
        printf("#                                                                        #\n");
        printf("# ##    ##    ####    ##     ##    ##              ##  ######  ##     ## #\n");
        printf("#  ##  ##    ##  ##   ##     ##    ##      ##      ##    ##    ###    ## #\n");
        printf("#   ####    ##    ##  ##     ##    ##     ####     ##    ##    ## #   ## #\n");
        printf("#    ##     ##    ##  ##     ##     ##   ##  ##   ##     ##    ##  #  ## #\n");
        printf("#    ##     ##    ##  ##     ##      ## ##    ## ##      ##    ##   # ## #\n");
        printf("#    ##      ##  ##    ##   ##        ###      ###       ##    ##    ### #\n");
        printf("#    ##       ####      #####          #        #      ######  ##     ## #\n");
        printf("#                                                                        #\n");
        printf("##########################################################################\n");
    }
}

int main4(void)
{
    int human, computer; // 1、2、3分别代表剪刀石头和布
    int result;
    int human_win = 0;
    int computer_win = 0;

    welcome();

    while (1)
    {
        human = get_human();
        computer = get_computer();

        // 用户输入0表示退出游戏
        if (human == 0)
        {
            break;
        }

        printf("你出");
        switch (human)
        {
        case 3:
            printf("剪刀，");
            break;
        case 6:
            printf("石头，");
            break;
        case 9:
            printf("布，");
            break;
        }

        printf("我出");
        switch (computer)
        {
        case 1:
            printf("剪刀，");
            break;
        case 2:
            printf("石头，");
            break;
        case 3:
            printf("布，");
            break;
        }

        result = human + computer;

        // 你出剪刀，电脑出布：3 + 3 == 6
        // 你出石头，电脑出剪刀：6 + 1 == 7
        // 你出布，电脑出石头：9 + 2 == 11
        // 以上三种情况算你赢～
        if (result == 6 || result == 7 || result == 11)
        {
            printf("你赢了！\n\n");
            human_win++;
        }
        else if (result == 5 || result == 9 || result == 10)
        {
            printf("我赢了！\n\n");
            computer_win++;
        }
        else
        {
            printf("咱打平！\n\n");
        }
    }

    // 打平也算人类赢
    if (human_win >= computer_win)
    {
        gameover(HUMANWIN);
    }
    else
    {
        gameover(COMPUTERWIN);
    }
}

#define MAX_LIMIT_MATRIX 100

void welcome(void);
int get_ins(void);
int *create_matrix(void);
void init_matrix(int *ptr);
void print_matrix(int *ptr);
void write_matrix(int *ptr);
void read_matrix(int *ptr);

void welcome(void)
{
    printf("\n============================\n");
    printf("* 欢迎使用该程序，指令如下 *\n");
    printf("* 1.生成一个 M*N 的矩阵    *\n");
    printf("* 2.初始化矩阵             *\n");
    printf("* 3.给矩阵中某个元素赋值   *\n");
    printf("* 4.读取矩阵中某个元素     *\n");
    printf("* 5.打印整个矩阵           *\n");
    printf("* 6.结束程序               *\n");
    printf("============================\n");
}

int get_ins(void)
{
    int ins;

    printf("\n请输入指令：");
    scanf("%d", &ins);

    while (ins < 1 || ins > 6)
    {
        printf("\n指令错误，请重新输入：");
        scanf("%d", &ins);
    }

    return ins;
}

int *create_matrix(void)
{
    int m, n;
    static int created = 0; // 用于判断是否已经创建过矩阵
    static int *ptr = NULL;

    if (created)
    {
        printf("矩阵已存在，是否需要重新创建？（Y/N）-> ");
        getchar(); // 清除缓冲区残留的换行符
        while (getchar() == 'N')
        {
            return ptr;
        }
    }

    printf("请输入新矩阵的规模（M*N）-> ");
    scanf("%d*%d", &m, &n);

    while (m < 1 || n < 1)
    {
        printf("规模太小，请重新输入：");
        scanf("%d*%d", &m, &n);
    }

    while (m > MAX_LIMIT_MATRIX || n > MAX_LIMIT_MATRIX)
    {
        printf("规模太大，请重新输入：");
        scanf("%d*%d", &m, &n);
    }

    // 虽然说是矩阵是二维数组，但在C语言中它的存放形式是“平铺”的
    // 这里用realloc，支持重新创建二维数组
    // 这里多申请了两个整形空间，用于存放矩阵的长和宽
    ptr = (int *)realloc(ptr, (m * n + 2) * sizeof(int));
    if (ptr == NULL)
    {
        printf("内存申请失败！\n");
        exit(1);
    }

    printf("%d*%d 的矩阵创建成功！\n", m, n);
    created = 1;

    // 将长和宽放在前两个元素中
    ptr[0] = m;
    ptr[1] = n;

    return ptr;
}

void init_matrix(int *ptr)
{
    int m = ptr[0];
    int n = ptr[1];
    int *matrix = ptr + 2;
    int num, i, j;

    if (ptr == NULL)
    {
        printf("未检测到矩阵，请先生成矩阵！\n");
        return;
    }

    printf("请输入一个数字：");
    scanf("%d", &num);

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            matrix[i * n + j] = num;
        }
    }
}

void print_matrix(int *ptr)
{
    int m = ptr[0];
    int n = ptr[1];
    int *matrix = ptr + 2;
    int i, j;

    if (ptr == NULL)
    {
        printf("未检测到矩阵，请先生成矩阵！\n");
        return;
    }

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("%d  ", matrix[i * n + j]);
        }
        putchar('\n');
    }
}

void write_matrix(int *ptr)
{
    int m = ptr[0];
    int n = ptr[1];
    int *matrix = ptr + 2;
    int num, x, y;

    if (ptr == NULL)
    {
        printf("未检测到矩阵，请先生成矩阵！\n");
        return;
    }

    printf("请输入要修改的位置（行,列）-> ");
    scanf("%d,%d", &x, &y);

    if (x > m || y > n || x < 1 || y < 1)
    {
        printf("坐标输入有误！\n");
        return;
    }

    printf("请输入一个数字：");
    scanf("%d", &num);

    matrix[(x - 1) * n + (y - 1)] = num;
}

void read_matrix(int *ptr)
{
    int m = ptr[0];
    int n = ptr[1];
    int *matrix = ptr + 2;
    int num, x, y;

    if (ptr == NULL)
    {
        printf("未检测到矩阵，请先生成矩阵！\n");
        return;
    }

    printf("请输入要读取的位置（行,列）-> ");
    scanf("%d,%d", &x, &y);

    if (x > m || y > n || x < 1 || y < 1)
    {
        printf("坐标输入有误！\n");
        return;
    }

    printf("第%d行，第%d列的数字是：%d\n", x, y, matrix[(x - 1) * n + (y - 1)]);
}

int main(void)
{
    int ins;
    int *ptr = NULL;

    welcome();

    while ((ins = get_ins()) != 6)
    {
        switch (ins)
        {
        case 1:
            ptr = create_matrix();
            break;
        case 2:
            init_matrix(ptr);
            break;
        case 3:
            write_matrix(ptr);
            break;
        case 4:
            read_matrix(ptr);
            break;
        case 5:
            print_matrix(ptr);
            break;
        }
    }

    printf("\n感谢使用本程序^_^\n\n");

    free(ptr);

    return 0;
}