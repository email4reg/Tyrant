#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


#define MAX 21
#define NUM 128

int func1()
{
    char str1[MAX], str2[MAX];
    int i = 0;
    unsigned n;

    printf("请输入第一个字符串:");
    while ((str1[i++] = getchar()) != '\n')
        ;
    printf("请输入第二个字符串:");
    i = 0;
    while ((str2[i++] = getchar()) != '\n')
        ;
    printf("请输入要比较的字符数:");
    scanf("%d", &n);

    for (i = 0; i < n; i++)
    {
        if (str1[i] - str2[i])
        {
            i++;
            break;
        }
    }

    printf("比较的结果是:%d\n", str1[i - 1] - str2[i - 1]);

    return 0;
}

int func2()
{
    char str[MAX]; // 包含结束符的21个字符
    int ch, space, i = 0;

    space = MAX - 1;

    printf("请输入一行文本:");
    while ((ch = getchar()) != '\n')
    {
        str[i++] = ch;
        if (i == MAX - 1)
        {
            break; // 字符数组的最后一个位置
        }
        if (ch == ' ')
        {
            space = i; // 记录最后一个空格的位置
        }
    }
    
    if (i >= MAX - 1)
    {
        str[space] = '\0';
    }
    else
    {
        str[i] = '\0';
    }
    
    printf("你输入的文本是:%s\n", str);

    return 0;
}

int func3()
{
    int ch, i, j = 0, max = 0;
    int input_num = 0;
    int ascii[NUM] = {0};
    char count[NUM] = "";

    printf("请输入英文文本:");

    while ((ch = getchar()) != '\n')
    {
        ascii[ch]++; // 字符对应的ASCII码加1
        input_num++;
    }

    for (i = 0; i < NUM; i++)
    {
        if (ascii[i])
        {
            count[j++] = i;
            if (ascii[i] > ascii[max])
            {
                max = i;
            }
        }
    }

    printf("你总共输入了%d个字符，其中不同的字符个数有%d个。\n", input_num, strlen(count));
    printf("它们是：%s\n", count);
    printf("出现次数最多的字符是\'%c\'，它总共出现了%d次。\n", max, ascii[max]);

    return 0;
}

int main()
{
    // func1();
    func2();
    // func3();

    return 0;
}