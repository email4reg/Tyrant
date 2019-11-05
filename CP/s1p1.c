#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


int func1()
{
    int i;
    unsigned sum = 0;
    
    for (i = 3; i < 1000; i++)
    {
        if ((i % 3 == 0) || (i % 5 == 0))
        {
            sum += i;
        }
    }

    printf("1000以下,属于3或5的倍数的和等于:%u\n", sum);

    return 0;
}

int func2()
{
    long a = 1, b = 2, c, sum = 0;

    do
    {
        if (!(b % 2))
        {
            sum += b;
        }
        c = a + b;
        a = b;
        b = c;
    } while (c < 4000000);

    printf("%ld\n", sum);

    return 0;
}

int func3(long n)
{
	int max = 0;

	for (int i = 2; i * i <= n; i++)
	{
		if (n % i == 0)
		{
            max = i;
        }
		while (n % i == 0)
		{
            n /= i;
        }
	}

	if (max < n)
    {
        max = n;
    }

	return max;
}

int fun4()
{
    int i, j, target, invert = 0, num = 998001; // 999 * 999

    for (; num > 10000; num--)
    {
        // 先求倒置数
        target = num;
        invert = 0;
        while (target)
        {
            invert = invert * 10 + target % 10;
            target = target / 10;
        }

        // 如果跟倒置数一致，说明该数是回文数
        if (invert == num)
        {
            for (i = 100; i < 1000; i++)
            {
                if (!(num % i) && (num / i >= 100) && (num / i < 1000))
                {
                    goto FINDIT;
                }
            }
        }
    }

FINDIT:
    printf("结果是%d == %d * %d\n", num, i, num / i);

    return 0;
}


int main()
{
    // func1();
    func2();
    // printf("最大质数因子为:%d\n", func3(600851475143));
    // func4();

    return 0;
}