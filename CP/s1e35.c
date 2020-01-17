#include <stdio.h>

#define MAX_NUM 64

int schedule[MAX_NUM + 1][MAX_NUM + 1];

int arrange(int begin, int num);

int arrange(int begin, int num)
{
    int i, j;

    if (num == 2)
    {
        schedule[begin][1] = begin;
        schedule[begin][2] = begin + 1;
        schedule[begin + 1][1] = begin + 1;
        schedule[begin + 1][2] = begin;
        return 0;
    }

    arrange(begin, num / 2);
    arrange(begin + num / 2, num / 2);

    for (i = begin + num / 2; i < begin + num; i++)
    {
        for (j = num / 2 + 1; j <= num; j++)
        {
            schedule[i][j] = schedule[i - num / 2][j - num / 2];
        }
    }

    for (i = begin; i < begin + num / 2; i++)
    {
        for (j = num / 2 + 1; j <= num; j++)
        {
            schedule[i][j] = schedule[i + num / 2][j - num / 2];
        }
    }
}

int main1(void)
{
    int num, i, j;

    printf("请输入参赛的队伍数量：");
    scanf("%d", &num);

    // 检查num是否2的N次方
    // 注意，这里是&，不是&&
    // &是按位与操作，1&1==1，0&1==0，0&0 == 0
    if (num & num - 1)
    {
        printf("参数队伍的数量必须是2的N次方！\n");
        return -1;
    }

    arrange(1, num);

    printf("编 号");

    for (i = 1; i < num; i++)
    {
        printf("\t第%d天", i);
    }

    putchar('\n');

    for (i = 1; i <= num; i++)
    {
        for (j = 1; j <= num; j++)
        {
            printf("%3d\t", schedule[i][j]);
        }
        putchar('\n');
    }

    return 0;
}


#define MAX_NUM 64

int schedule[MAX_NUM + 1][MAX_NUM + 1];

void getInput(char name[][128], int num);
int arrange(int begin, int num);

void getInput(char name[][128], int num)
{
    int i;

    for (i = 0; i < num; i++)
    {
        printf("请输入第%d个队伍的名字：", i + 1);
        scanf("%s", name[i]);
        getchar();
    }
}

int arrange(int begin, int num)
{
    int i, j;

    if (num == 2)
    {
        schedule[begin][1] = begin;
        schedule[begin][2] = begin + 1;
        schedule[begin + 1][1] = begin + 1;
        schedule[begin + 1][2] = begin;
        return 0;
    }

    arrange(begin, num / 2);
    arrange(begin + num / 2, num / 2);

    for (i = begin + num / 2; i < begin + num; i++)
    {
        for (j = num / 2 + 1; j <= num; j++)
        {
            schedule[i][j] = schedule[i - num / 2][j - num / 2];
        }
    }

    for (i = begin; i < begin + num / 2; i++)
    {
        for (j = num / 2 + 1; j <= num; j++)
        {
            schedule[i][j] = schedule[i + num / 2][j - num / 2];
        }
    }
}

int main(void)
{
    int num, i, j;

    printf("请输入参赛的队伍数量：");
    scanf("%d", &num);

    // 检查num是否2的N次方
    // 注意，这里是&，不是&&
    // &是按位与操作，1&1==1，0&1==0，0&0 == 0
    if (num & num - 1)
    {
        printf("参数队伍的数量必须是2的N次方！\n");
        return -1;
    }

    char name[num][128];

    getInput(name, num);

    arrange(1, num);

    printf("\n比赛安排如下：\n");
    printf("队 伍");

    for (i = 1; i < num; i++)
    {
        printf("\t第%d天", i);
    }

    putchar('\n');

    for (i = 1; i <= num; i++)
    {
        for (j = 1; j <= num; j++)
        {
            printf("%s\t", name[schedule[i][j] - 1]);
        }
        putchar('\n');
    }

    return 0;
}