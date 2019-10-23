#include <stdio.h>
#include <stdlib.h>
#include <math.h>



int func1()
{
    int count = 0;

    printf("请输入一串英文字符:");
    
    while (getchar() != '\n')
    {
        count += 1;
    }
    printf("你总共输入了%d字符!\n", count);

    return 0;
}

int func2()
{
    int count = 0;
    char ch;

    printf("请输入一行英文句子:");
    while ((ch = getchar()) != '\n')
    {
        if (ch >= 'A' && ch <= 'Z')
        {
            count += 1;
        }
    }
    printf("你总共输入了%d个大写字母\n", count);

    return 0;
}

int func3()
{
    char ch;

    printf("请输入一行英文句子:");

    while ((ch = getchar()) != '\n')
    {
        if (ch >= 'A' && ch <= 'Z')
        {
            ch = ch - 'A' + 'a'; // 大小写转换的方法
        }
        else if (ch >= 'a' && ch <= 'z')
        {
            ch = ch - 'a' + 'A';
        }
        putchar(ch);
    }

    putchar('\n');

    return 0;
}

int func4()
{
    int ch;
    int num = 0;
    
    printf("请输入待转换待字符串:");

    do
    {
        ch = getchar();

        if (ch >= '0' && ch <= '9')
        {
            num = 10 * num + (ch - '0');
        }
        else
        {
            if (num)
            {
                break; // 如果已有数字,则退出循环
            }
        }
        
    } while (ch != '\n');

    printf("结果是:%d\n",num);

    return 0;
}

int func5()
{
    char ch;
    long long num = 0;
    long long temp;
    int is_overflow = 0;

    const int max_int = pow(2,sizeof(int) * 8) / 2 - 1; // const 防止常量被修改
    const int min_int = pow(2, sizeof(int) * 8) / 2 * (-1);

    printf("请输入待转换的字符串:");

    do
    {
        ch = getchar();
        if (ch >= '0' && ch <= '9')
        {
            temp = 10 * num + (ch - '0');
            if (temp > max_int || temp < min_int)
            {
                is_overflow = 1;
                break;
            }
        }
        else
        {
            if (num)
            {
                break;
            }
        }       
    } while (ch != '\n');
    
    if (is_overflow)
    {
        printf("数值超出范围,结果未定义!\n");
    }
    else
    {
        if (!num)
        {
            printf('并未找到任何数值!\n');
        }
        else
        {
            printf("结果是:%d\n",num);
        }   
    }
    
    return 0;
}


int main()
{
    // func1();
    // func2();
    // func3();
    // func4();
    func5();

    return 0;
}