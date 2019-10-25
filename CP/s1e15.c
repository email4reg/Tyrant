#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int func1()
{
    char ch;
    int count_a = 0;
    int count_e = 0;
    int count_i = 0;
    int count_o = 0;
    int count_u = 0;
    int sum = 0;

    printf("请输入一个英文句子:");
    while ((ch = getchar()) != '\n')
    {
        if (ch == 'a' || ch == 'A') // 可以改写成switch语句
        {
            count_a++;
        }
        else if (ch == 'e' || ch == 'E')
        {
            count_e++;
        }
        else if (ch == 'i' || ch == 'I')
        {
            count_i++;
        }
        else if (ch == 'o' || ch == 'O')
        {
            count_o++;
        }
        else if (ch == 'u' || ch == 'U')
        {
            count_u++;
        }
        else
        {
            ;
        }  
    }

    sum = count_a + count_e + count_i + count_o + count_u;

    printf("您输入的句子中,包含元音字母%d个!\n",sum);
    printf("其中,a(%d),e(%d),i(%d),o(%d),u(%d)\n",count_a,count_e,count_i,count_o,count_u);

    return 0;
}

int func2() // 与小甲鱼不同
{
    int count = 2; // 已知2，3是素数，4不是
    int num = 10000;

    for (int i = 5; i <= num; i++)
    {
        for (int j = 2; j <= i / 2; j++) // 实际上，满足2~sqrt(i)的平方根即可
        {
            if (i % j)
            {
                continue;
            }
            else
            {
                count--;
                break;
            }
        }
        count++;
    }
    
    printf("%d以内共有%d个素数(质数)\n",num,count);

    return 0;
}

int func3()
{
    char ch;

    printf("请输入明文:");
    while ((ch = getchar()) != '\n')
    {
        if (ch >= 'A' && ch <= 'Z')
        {
            putchar('A' + (ch - 'A' + 3) % 26);
            continue;

        }
        else if (ch >= 'a' && ch <= 'z')
        {
            putchar('a' + (ch - 'a' + 3) % 26);
            continue;
        }
        putchar(ch);
    }
    
    putchar('\n');

    return 0;
}

int main()
{
    // func1();
    // func2();
    func3();

    return 0;
}