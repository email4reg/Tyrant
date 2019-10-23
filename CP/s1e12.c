#include <stdio.h>
#include <math.h>

/*为了方便调用，我们为控制左右电机, 
前进后退的数字引脚进行了宏定义*/
#define LEFT_MOTO_GO 8
#define LEFT_MOTO_BACK 9
#define RIGHT_MOTO_GO 10
#define RIGHT_MOTO_BACK 11


int func1()
{
    float op1,op2;
    char ch;

    printf("请输入一个式子:\n");
    scanf("%f %c %f",&op1,&ch,&op2);

    switch (ch)
    {
        case '-': printf("结果是:%.2f\n", op1 - op2);break;
        case '+': printf("结果是:%.2f\n", op1 + op2);break;
        case '*': printf("结果是:%.2f\n", op1 * op2);break;
        case '/': 
            if (op2 == 0)
            {
                printf("很遗憾,除数不能为0\n");
                break;
            }
            else
            {
                printf("结果是:%.2f\n", op1 / op2);
                break;
            }
        default: printf("请输入有效式子形式!\n");
    }
    
    return 0;
}

void loop()
{
    char ch; // 用于接受命令

    printf("输入命令:");
    scanf("%c",&ch);

    switch (ch)
    {
        case 'g':
        {
            digitalWrite(LEFT_MOTO_GO, HIGH);
            digitalWrite(LEFT_MOTO_BACK, LOW);
            digitalWrite(RIGHT_MOTO_GO, HIGH);
            digitalWrite(RIGHT_MOTO_BACK, LOW);
            break;
        }
        case 'b':
        {
            digitalWrite(LEFT_MOTO_BACK,HIGH);
            digitalWrite(LEFT_MOTO_GO,LOW);
            digitalWrite(RIGHT_MOTO_BACK, HIGH);
            digitalWrite(RIGHT_MOTO_GO,LOW);
            break;
        }
        case 'r':
        {
            digitalWrite(LEFT_MOTO_GO,HIGH);
            digitalWrite(LEFT_MOTO_BACK, LOW);
            digitalWrite(RIGHT_MOTO_BACK, LOW);
            digitalWrite(RIGHT_MOTO_GO, LOW);
            break;
        }
        case 'l':
        {
            digitalWrite(LEFT_MOTO_GO, LOW);
            digitalWrite(LEFT_MOTO_BACK, LOW);
            digitalWrite(RIGHT_MOTO_GO, HIGH);
            digitalWrite(RIGHT_MOTO_BACK, LOW);
            break;
        }
        default:
        {
            digitalWrite(LEFT_MOTO_GO, LOW);
            digitalWrite(LEFT_MOTO_BACK, LOW);
            digitalWrite(RIGHT_MOTO_GO, LOW);
            digitalWrite(RIGHT_MOTO_BACK, LOW);
            break;
        }
    }
    // ...省略部分代码...
}

int main()
{
    func1();

    return 0;
}