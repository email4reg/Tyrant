#include <stdio.h>
#include <math.h>


int func1()
{
    float fah,cels;

    printf("请输入华氏度:\n");
    scanf("%f",&fah);

    cels = (fah - 32) * 5 / 9;
    printf("转换为摄氏度是:%.2f\n",cels);

    return 0;
}

int func2()
{
    char name[256];
    float height,weight;

    printf("请输入您的姓名:");
    scanf("%s",name);

    printf("请输入您的身高(cm):");
    scanf("%f", &height);

    printf("请输入您的体重(kg):");
    scanf("%f", &weight);

    printf("正在为您转换==>>>>>>\n");
    printf("%s的身高是%.2f(in),体重是%(lb).2f\n",name,height / 2.54,weight / 0.453);

    return 0;
}

int main()
{
    // func1();
    func2();

    return 0;
}