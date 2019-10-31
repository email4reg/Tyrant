#include <stdio.h>
#include<math.h>

#define MAIZI 25000

int func1()
{
    int result;

    result = pow(1,2) + pow(2,3) + pow(3,4) + pow(4,5) + pow(5,6);
    printf("值为:%d\n",result);

    return 0;
}

int func2()
{
    int i;
    unsigned long long int sum = 0;

    for (i=0;i<64;i++)
    {
        sum += (int)pow(2, i); // pow返回的是double，需要强制转换成int
    }

    printf("舍罕王应该给予达依尔%llu粒麦子!\n如果每25000粒麦子为1kg,那么应该给%llukg麦子\n", sum, sum / MAIZI);

    return 0;
}

int main()
{
    // func1();
    func2();
    
    return 0;
}
