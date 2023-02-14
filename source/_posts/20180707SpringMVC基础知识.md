---
layout: post
title: "SpringMVC基础知识"
date: 2018-07-07 10:36
toc: true
comments: true
categories: 技术学习
tags:
	- web 
---

#### 概述

咱们开发服务器端程序，一般都基于两种形式，一种C/S架构程序，一种B/S架构程序

使用Java语言基本上都是开发B/S架构的程序，B/S架构又分成了三层架构

SpringMVC属于表现层框架

<!--more-->

##### 三层架构

```
表现层：WEB层，它负责接收客户端请求，向客户端响应结果,用来和客户端进行数据交互的。表现层一般会采用MVC的设计模型，表现层包括展示层和控制层：控制层负责接收请求，展示层负责结果的展示。

业务层：处理公司具体的业务逻辑的

持久层：用来操作数据库的,对数据库表进行曾删改查。
```

##### MVC模型

MVC全名是Model View Controller 模型视图控制器，每个部分各司其职。

```
Model：数据模型，JavaBean的类，用来进行数据封装。
View：指JSP、HTML用来展示数据给用户
Controller：用来接收用户的请求，整个流程的控制器。用来进行数据校验等
```

![springmvc执行流程原理](https://cdn.jsdelivr.net/gh/siyuanzhou/pic@master/pic/2018-07-07-SpringMVC%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/springmvc%E6%89%A7%E8%A1%8C%E6%B5%81%E7%A8%8B%E5%8E%9F%E7%90%86.jpg)

```
DispatcherServlet用户请求到达前端控制器，它就相当于 mvc 模式中的 c，dispatcherServlet 是整个流程控制的中心，由它调用其它组件处理用户的请求，dispatcherServlet 的存在降低了组件之间的耦合性。

HandlerMapping 负责根据用户请求找到 Handler 即处理器，SpringMVC 提供了不同的映射器实现不同的映射方式，例如：配置文件方式，实现接口方式，注解方式等。

Handler 它就是我们开发中要编写的具体业务控制器。由 DispatcherServlet 把用户请求转发到 Handler。由
Handler 对具体的用户请求进行处理。

HandlerAdapter 对处理器进行执行，这是适配器模式的应用，通过扩展适配器可以对更多类型的处理器进行执行

View Resolver 负责将处理结果生成 View 视图，View Resolver 首先根据逻辑视图名解析成物理视图名即具体的页面地址，再生成 View 视图对象，最后对 View 进行渲染将处理结果通过页面展示给用户。

SpringMVC 框架提供了很多的View视图类型的支持，包括：jstlView、freemarkerView、pdfView等。最常用的视图就是jsp。一般情况下需要通过页面标签或页面模版技术将模型数据通过页面展示给用户
```

##### SpringMVC

```
SpringMVC 是一种基于 Java 的实现 MVC 设计模型的请求驱动类型的轻量级 Web 框架
它通过一套注解，让一个简单的 Java 类成为处理请求的控制器，而无须实现任何接口
还有比如RESTful风格的支持、简单的文件上传、约定大于配置的契约式编程支持、基于注解的零配置支持

SpringMVC与Struts2区别：
    Spring MVC 的入口是 Servlet, 而 Struts2 是 Filter
    Spring MVC 是基于方法设计的，而Struts2是基于类，Struts2每次执行都会创建一个动作类。所以 Spring MVC 会稍微比 Struts2 快些。
    Spring MVC 使用更加简洁,同时还支持 JSR303, 处理 ajax 的请求更方便
    (JSR303 是一套 JavaBean 参数校验的标准，它定义了很多常用的校验注解，我们可以直接将这些注解加在我们 JavaBean 的属性上面，就可以在需要校验的时候进行校验了。)
```

![image-20200309183433709](https://cdn.jsdelivr.net/gh/siyuanzhou/pic@master/pic/2018-07-07-SpringMVC%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/image-20200309183433709.png)

##### 入门案例

![image-20200309185414653](https://cdn.jsdelivr.net/gh/siyuanzhou/pic@master/pic/2018-07-07-SpringMVC%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/image-20200309185414653.png)

1 依赖文件pom.xml

```xml
<!-- 版本锁定 -->
<properties>
    <spring.version>5.0.2.RELEASE</spring.version>
</properties>
<dependencies>
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-context</artifactId>
        <version>${spring.version}</version>
    </dependency>
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-web</artifactId>
        <version>${spring.version}</version>
    </dependency>
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-webmvc</artifactId>
        <version>${spring.version}</version>
    </dependency>
    <dependency>
        <groupId>javax.servlet</groupId>
        <artifactId>servlet-api</artifactId>
        <version>2.5</version>
        <scope>provided</scope>
    </dependency>
    <dependency>
        <groupId>javax.servlet.jsp</groupId>
        <artifactId>jsp-api</artifactId>
        <version>2.0</version>
        <scope>provided</scope>
    </dependency>
</dependencies>
```

2 在web.xml配置文件中配置核心控制器DispatcherServlet

```xml
<!DOCTYPE web-app PUBLIC
 "-//Sun Microsystems, Inc.//DTD Web Application 2.3//EN"
 "http://java.sun.com/dtd/web-app_2_3.dtd" >

<!-- SpringMVC的核心控制器 -->
<web-app>
  <display-name>Archetype Created Web Application</display-name>
  <servlet>
    <servlet-name>dispatcherServlet</servlet-name>
    <servlet-class>org.springframework.web.servlet.DispatcherServlet</servlet-class>
    <!-- 配置Servlet的初始化参数，读取springmvc的配置文件，创建spring容器 -->
    <init-param>
      <param-name>contextConfigLocation</param-name>
      <param-value>classpath:springmvc.xml</param-value>
    </init-param>
    <!-- 配置servlet启动时加载对象 -->
    <load-on-startup>1</load-on-startup>
  </servlet>
  <servlet-mapping>
    <servlet-name>dispatcherServlet</servlet-name>
    <url-pattern>/</url-pattern>
  </servlet-mapping>
</web-app>
```

3 在resources文件夹内编写springmvc.xml的配置文件

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:mvc="http://www.springframework.org/schema/mvc"
       xmlns:context="http://www.springframework.org/schema/context"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="
        http://www.springframework.org/schema/beans
        http://www.springframework.org/schema/beans/spring-beans.xsd
        http://www.springframework.org/schema/mvc
        http://www.springframework.org/schema/mvc/spring-mvc.xsd
        http://www.springframework.org/schema/context
        http://www.springframework.org/schema/context/spring-context.xsd">
    <!--
    在 SpringMVC 的各个组件中，处理器映射器、处理器适配器、视图解析器称为 SpringMVC 的三大组件。
    使用 <mvc:annotation-driven>自动加载RequestMappingHandlerMapping(处理映射器)和
    RequestMappingHandlerAdapter(处理适配器) 
    可用在SpringMVC.xml配置件中使用<mvc:annotation-driven>替代注解处理器和适配器的配置。
    -->
    <!-- 配置spring创建容器时要扫描的包 -->
    <context:component-scan base-package="site.newvalue.controller"></context:component-scan>
    <!-- 配置视图解析器 -->
    <bean id="internalResourceViewResolver" class="org.springframework.web.servlet.view.InternalResourceViewResolver">
        <property name="prefix" value="/WEB-INF/pages/"></property>
        <property name="suffix" value=".jsp"></property>
    </bean>
    <!-- 配置spring开启注解mvc的支持  -->
    <mvc:annotation-driven></mvc:annotation-driven>
    <!--在 springmvc 的配置文件中可以配置，静态资源不过滤：-->
    <!-- location 表示路径，mapping 表示文件，**表示该目录下的文件以及子目录的文件 -->
    <mvc:resources location="/css/" mapping="/css/**"/>
    <mvc:resources location="/images/" mapping="/images/**"/>
    <mvc:resources location="/scripts/" mapping="/javascript/**"/>
</beans>
```

4 index.jsp

```html
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
<body>
    <h3>入門程序</h3>
    <a href="hello">入门程序</a>
</body>
</html>

//在WEB-INF目录下创建pages文件夹，编写success.jsp的成功页面
<body>
<h3>入门成功！！</h3>
</body>
```

5 HelloController

```java
@Controller
public class HelloController {
    @RequestMapping(path = "/hello")
    public  String sayHello()
    {
        System.out.println("hello");
        return  "success";
    }
}
```

##### 入门案例的执行流程

```java
当启动Tomcat服务器的时候，因为配置了load-on-startup标签，所以会创建DispatcherServlet对象，就会加载springmvc.xml配置文件

开启了注解扫描，那么HelloController对象就会被创建

从index.jsp发送请求，请求会先到达DispatcherServlet核心控制器，根据配置@RequestMapping注解找到执行的具体方法

根据执行方法的返回值，再根据配置的视图解析器，去指定的目录下查找指定名称的JSP文件

Tomcat服务器渲染页面，做出响应
```

#### 请求参数

#####  RequestMapping注解

```java
RequestMapping注解的作用是建立请求URL和处理方法之间的对应关系
RequestMapping注解可以作用在方法和类上
作用在类上：第一级的访问目录
作用在方法上：第二级的访问目录
细节：路径可以不编写/表示应用的根目录开始
细节：${ pageContext.request.contextPath }也可以省略不写，但是路径上不能写 /

RequestMapping的属性
    path  指定请求路径的url
    value value属性和path属性是一样的
    mthod 指定该方法的请求方式
    params 指定限制请求参数的条件
    headers 发送的请求中必须包含的请求头
```

##### 请求参数的绑定

```java
绑定机制
    表单提交的数据都是k=v格式的 username=haha&password=123
    SpringMVC的参数绑定过程是把表单提交的请求参数，与控制器中方法参数进行绑定
    要求：提交表单的name和参数的名称是相同的
        <a href="account/findAccount?accountId=10">查询账户</a>
        中请求参数是：accountId=10
        //查询账户
        @RequestMapping("/findAccount")
        public String findAccount(Integer accountId) {
            System.out.println("查询了账户。。。。"+accountId);
            return "success";
        }
支持的数据类型
    基本数据类型和字符串类型
    实体类型（JavaBean）
    集合数据类型（List、map集合等）
    
基本数据类型和字符串类型
	提交表单的name和参数的名称是相同的，区分大小写
实体类型（JavaBean）
    提交表单的name和JavaBean中的属性名称需要一致
    若JavaBean类中包含其他引用类型，那么表单name属性需要编写成：对象.属性，如：address.name
        <!-- pojo 类型演示 -->
        public class Account implements Serializable {
            private String name;
            private Float money;
            private Address address;//provinceName,cityName
            //getters and setters
        }
        <form action="account/saveAccount" method="post">
            账户名称：<input type="text" name="name" ><br/>
            账户金额：<input type="text" name="money" ><br/>
            账户省份：<input type="text" name="address.provinceName" ><br/>
            账户城市：<input type="text" name="address.cityName" ><br/>
            <input type="submit" value=" 保存 ">
        </form>
        @RequestMapping("/saveAccount")
        public String saveAccount(Account account) {
            System.out.println("保存了账户。。。。"+account);
            return "success";
        }
给集合属性数据封装
	JSP页面编写方式：list[0].属性
        账户 1 名称：<input type="text" name="accounts[0].name" ><br/>
        账户 1 金额：<input type="text" name="accounts[0].money" ><br/>
        账户 2 名称：<input type="text" name="accounts[1].name" ><br/>
        账户 2 金额：<input type="text" name="accounts[1].money" ><br/>
```

##### 请求参数中文乱码

在web.xml中配置Spring提供的过滤器类

```xml
<!-- 配置 springMVC 编码过滤器，解决中文乱码 -->
<filter>
    <filter-name>CharacterEncodingFilter</filter-name>
    <filter-class>
        org.springframework.web.filter.CharacterEncodingFilter
    </filter-class>
    <!-- 设置过滤器中的属性值 -->
    <init-param>
        <param-name>encoding</param-name>
        <param-value>UTF-8</param-value>
    </init-param>
    <!-- 启动过滤器 -->
    <init-param>
        <param-name>forceEncoding</param-name>
        <param-value>true</param-value>
    </init-param>
</filter>
<!-- 过滤所有请求 -->
<filter-mapping>
    <filter-name>CharacterEncodingFilter</filter-name>
    <url-pattern>/*</url-pattern>
</filter-mapping>
```

```xml
tomacat对GET和POST请求处理方式是不同，GET请求编码问题改tomcat的server.xml
<Connector connectionTimeout="20000" port="8080"
protocol="HTTP/1.1" redirectPort="8443"/>
改为：
<Connector connectionTimeout="20000" port="8080"
protocol="HTTP/1.1" redirectPort="8443"
useBodyEncodingForURI="true"/>
如果遇到 ajax 请求仍然乱码，请把：useBodyEncodingForURI="true"改为 URIEncoding="UTF-8"
```

##### 自定义类型转换器

```
Spring框架内部会默认进行数据类型转换
	表单提交的任何数据类型全部都是字符串类型，但是后台定义Integer类型，数据也可以封装上
```

如果想自定义数据类型转换，可以实现Converter的接口，将String转Date

```java
<!-- 特殊情况之：类型转换问题 -->
<a href="account/deleteAccount?date=2018-01-01">根据日期删除账户</a>
//自定义数据类型转换类
public class StringToDate implements Converter<String, Date> {
    public Date convert(String s) {
        if (s == null) {
            throw new RuntimeException("请输入日期数据");
        } else {
            DateFormat df = new SimpleDateFormat("yyyy-MM-dd");
            Date date = null;
            try {
                date = df.parse(s);
            } catch (ParseException e) {
                throw new RuntimeException("日期准换错误");
            }
            return date;
        }
    }
}
```

注册自定义类型转换器，在springmvc.xml配置文件中编写配置

```xml
<!-- 注册自定义类型转换器 -->
<bean id="conversionService" class="org.springframework.context.support.ConversionServiceFactoryBean">
    <property name="converters">
        <set>
            <bean class="site.newvalue.utils.StringToDate"></bean>
        </set>
    </property>
</bean>
<!-- 配置spring开启注解mvc的支持 -->
<mvc:annotation-driven conversion-service="conversionService"></mvc:annotation-driven>
```

##### ServletAPI做控制器参数

只需要在控制器的方法参数定义HttpServletRequest和HttpServletResponse对象

```java
//只需要在控制器的方法参数定义HttpServletRequest和HttpServletResponse对象
@RequestMapping("/testServletAPI")
public String testServletAPI(HttpServletRequest request,
        HttpServletResponse response,
        HttpSession session) {
        System.out.println(request);
        return "success";
}
```

#### 常用注解

##### RequestParam注解

```
作用：把请求中的指定名称的参数传递给控制器中的形参赋值
属性
    1. value：请求参数中的名称
    2. required：请求参数中是否必须提供此参数，默认值是true，必须提供
```

```java
@RequestMapping(path="/hello")
public String sayHello(@RequestParam(value="username",required=false)String name) {
    System.out.println(name);
    return "success";
}
```

##### RequestBody注解

```
作用：用于获取请求体（所有form内容）的内容（注意：get方法不可以）
	直接使用得到是 key=value&key=value...结构的数据
属性： required：是否必须有请求体，默认值是true
```

```java
<!-- request body 注解 -->
<form action="useRequestBody" method="post">
    用户名称：<input type="text" name="username" ><br/>
    用户密码：<input type="password" name="password" ><br/>
    用户年龄：<input type="text" name="age" ><br/>
    <input type="submit" value=" 保存 ">
</form>
@RequestMapping("/useRequestBody")
public String  useRequestBody(@RequestBody(required=false) String body){
    System.out.println(body);
    return "success";
}
```

#####  PathVariable注解

```java
作用：拥有绑定url中的占位符的。例如：url中有/delete/{id}，{id}就是占位符

属性value：指定url中的占位符名称
<a href="user/hello/1">入门案例</a>
@RequestMapping(path="/hello/{id}")
public String sayHello(@PathVariable(value="id") String id) {
    System.out.println(id);
    return "success";
}
```

补充：

```java
Restful风格的URL：请求路径一样，可以根据不同的请求方式去执行后台的不同方法
    restful风格的URL优点:结构清晰、符合标准、易于理解、扩展方便
    
    资源（ Resources）：网络上的一个实体，或者说是网络上的一个具体信息。
    它可以是一段文本、一张图片、一首歌曲、一种服务，总之就是一个具体的存在。可以用一个 URI（统一资源定位符）指向它，每种资源对应一个特定的 URI 。
    要获取这个资源，访问它的 URI 就可以，因此 URI 即为每一个资源的独一无二的识别符。

    表现层（ Representation）：把资源具体呈现出来的形式，叫做它的表现层 （ Representation）。
    比如，文本可以用 txt 格式表现，也可以用 HTML 格式、XML 格式、JSON 格式表现，甚至可以采用二进制格式。

    状态转化（State Transfer）：每发出一个请求，就代表了客户端和服务器的一次交互过程。
    HTTP协议，是一个无状态协议，即所有的状态都保存在服务器端。因此，如果客户端想要操作服务器，必须通过某种手段，让服务器端发生 “状态转化 ”（ State Tran sfer）。
    
    restful  的示例：
    /account/1 HTTP  GET ：  得到 id = 1 的 account
    /account/1 HTTP  DELETE： 删除 id = 1 的 account
    /account/1 HTTP  PUT：  更新 id = 1 的 account
    /account HTTP  POST：  新增 account

由于浏览器 form 表单只支持 GET 与 POST 请求，而 DELETE、PUT 等 method 并不支持，Spring3.0 添加了一个过滤器 HiddentHttpMethodFilter，可以将浏览器请求改为指定的请求方式，发送给我们的控制器方法，使得支持 GET、POST、PUT与 DELETE 请求。
使用方法：
    第一步：在 web.xml 中配置该过滤器。
    第二步：请求方式必须使用 post 请求。
    第三步：按照要求提供_method 请求参数，该参数的取值就是我们需要的请求方式。   
<!-- 删除 -->
<form action="springmvc/testRestDELETE/1" method="post">
    <input type="hidden" name="_method" value="DELETE">
    <input type="submit" value=" 删除 ">
</form>
@RequestMapping(value="/testRestDELETE/{id}",method=RequestMethod.DELETE)
public String  testRestfulURLDELETE(@PathVariable("id")Integer id){
    System.out.println("rest delete "+id);
    return "success";
}
```

##### RequestHeader注解

```java
作用：获取指定请求头的值
属性value：请求头的名称(不常用)
@RequestMapping(path="/hello")
public String sayHello(@RequestHeader(value="Accept") String header) {
    System.out.println(header);
    return "success";
}
```

##### CookieValue注解

```java
作用：用于获取指定cookie的名称的值并传入控制器方法参数。
属性，value：cookie的名称
@RequestMapping(path="/hello")
public String sayHello(@CookieValue(value="JSESSIONID") String cookieValue) {
    System.out.println(cookieValue);
    return "success";
}
```

##### ModelAttribute注解

```
作用
    1. 出现在方法上：表示当前方法会在控制器方法执行前先执行
    2. 出现在参数上：获取指定的数据给参数赋值
value：用于获取数据的 key。key 可以是 POJO 的属性名称，也可以是 map 结构的 key。
使用场景：当提交表单数据不是完整的实体数据时，保证没有提交的字段使用数据库原来的数据。
	如：我们在编辑一个用户时，用户有一个创建信息字段，该字段的值是不允许被修改的。
	在提交表单数据是肯定没有此字段内容，一旦更新会把该字段内容置为 null，此时就可以使用此注解解决问题。
```

修饰的方法有返回值

```java
<a href="updateUser?name=test">测试 modelattribute</a>
@ModelAttribute
public User showUser(String name) {
	System.out.println("showUser执行了...");
    // 模拟从数据库中查询对象
    User user = new User();
    user.setName("哈哈");
    user.setPassword("123");
    user.setMoney(100d);
    return user;
}
@RequestMapping(path="/updateUser")
public String updateUser(User user) {
    System.out.println(user);
    return "success";
}
```

修饰的方法无返回值

```java
<a href="updateUser?name=test">测试 modelattribute</a>
@ModelAttribute
public void showUser(String name,Map<String, User> map) {
    System.out.println("showUser执行了...");
    // 模拟从数据库中查询对象
    User user = new User();
    user.setName(name);
    user.setPassword("123");
    user.setMoney(100);
    map.put("abc", user);
}
@RequestMapping(path="/updateUser")
public String updateUser(@ModelAttribute(value="abc") User user) {
    System.out.println(user);
    return "success";
}
```

##### SessionAttributes注解

```java
作用：用于多次执行控制器方法间的参数共享
属性，value：指定存入属性的名称
<!-- SessionAttribute 注解的使用 -->
<a href="user/user">存入 SessionAttribute</a>
<a href="user/find">取出 SessionAttribute</a>
<a href="user/delete">清除 SessionAttribute</a>
@Controller
@RequestMapping(path="/user")
@SessionAttributes(value={"username","password","age"},types={String.class,Integer.class})
public class SessionContorller {
    //向session中存入值
    @RequestMapping(path="/user")
    public String save(Model model) {
        System.out.println("向session域中保存数据");
        model.addAttribute("username", "root");
        model.addAttribute("password", "123");
        model.addAttribute("age", 20);
        return "success";
    }
    //从session中获取值
    @RequestMapping(path="/find")
    public String find(ModelMap modelMap) {
        String username = (String) modelMap.get("username");
        String password = (String) modelMap.get("password");
        Integer age = (Integer) modelMap.get("age");
        System.out.println(username + " : "+password +" : "+age);
        return "success";
    }
    //清除值
    @RequestMapping(path="/delete")
    public String delete(SessionStatus status) {
        status.setComplete();
        return "success";
    }
}
```

#### 响应数据和结果视图

##### 返回字符串

Controller方法返回字符串可以指定逻辑视图的名称，根据视图解析器为物理视图的地址

```java
////指定逻辑视图名，经过视图解析器解析为jsp物理路径：/WEB-INF/pages/success.jsp
@RequestMapping("/getUser")
public String getUser(Model model)
{return "success";}
```

##### 返回值为void

如果控制器的方法返回值编写成void，执行程序报404的异常，默认查找JSP页面没有找到。默认会跳转到@RequestMapping(value="/initUpdate") initUpdate.jsp的页面

可以使用请求转发或者重定向跳转到指定的页面

```java
@RequestMapping("/getVoid")
public void getVoid(HttpServletRequest request, HttpServletResponse response) throws IOException, ServletException {
    System.out.println("请求转发");
    //        request.getRequestDispatcher("/WEB-INF/pages/success.jsp").forward(request,response);
    System.out.println("请求重定向");
    //请求不到WEB-INF里的东西
    //response.sendRedirect(request.getContextPath()+"/index.jsp");
    //直接响应
    response.setCharacterEncoding("UTF-8");
    response.setContentType("text/html;charset=utf-8");
    response.getWriter().println("你好");
}
```

##### 返回值为ModelAndView对象

ModelAndView对象是Spring提供的一个对象，可以用来调整具体的JSP视图

```java
@RequestMapping("/testModelAndView")
public ModelAndView testModelAndView(){
    // 创建ModelAndView对象
    ModelAndView mv = new ModelAndView();
    // 模拟从数据库中查询出User对象
    User user = new User();
    user.setName("小凤");
    user.setPassword("456");
    // 把user对象存储到mv对象中，也会把user对象存入到request对象
    mv.addObject("user",user);
    // 跳转到哪个页面
    mv.setViewName("success");
    return mv;
}
响应的 jsp  代码：
<%@page language="java"  contentType="text/html;  charset=UTF-8"
           pageEncoding="UTF-8" isELIgnored="false"%>
<!DOCTYPE  html  PUBLIC  "-//W3C//DTD  HTML  4.01  Transitional//EN"
"http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
    <title>执行成功</title>
</head>
<body>
执行成功！
${requestScope.user.name}
</body>
</html>
```

##### 转发和重定向

```java
@RequestMapping("/testForwardOrRedirect")
public String testForwardOrRedirect()
{
    System.out.println("testForwardOrRedirect执行");
    //forward:转发的JSP路径，不走视图解析器了，所以需要编写完整的路径
    //return "forward:/WEB-INF/pages/success.jsp";
    return "redirect: /index.jsp";
}

//如果是重定向到 jsp 页面，则 jsp 页面不能写在 WEB-INF 目录中，否则无法找到。
@RequestMapping("/count")
public String count() throws Exception {
	System.out.println("count方法执行了...");
    return "redirect:/add.jsp";
    // return "redirect:/user/findAll";
}
```

##### ResponseBody响应json数据

jsp中使用Ajax进行异步调用。

```xml
<script src="js/jquery.min.js"></script>
<script>
    $(function () {
        $("#btn").click(function () {
            $.ajax({
                url:"testAjax",
                contentType:"application/json;charset=UTF-8",
                data:'{"name":"zsy","password":"123"}',
                datatype:"json",
                type:"post",
                success:function (data) {
                    alert(data);
                    alert(data.name);
                }
            });
        });
    });
</script>
<button id="btn">发送ajax请求</button>
```

```java
@RequestMapping("/testAjax")
//第一个使用@ResponseBody注解把JavaBean对象转换成json字符串，直接响应
//第二个使用@RequestBody注解把json的字符串转换成JavaBean的对象
public @ResponseBody User testAjax(@RequestBody User user){
    // 客户端发送ajax的请求，传的是json字符串，后端把json字符串封装到user对象中
    System.out.println(user);
    // 做响应，模拟查询数据库
    user.setName("haha");
    return user;
}
```

json字符串和JavaBean对象互相转换的过程中，Springmvc 默认用 MappingJacksonHttpMessageConverter 对 json 数据进行转换，需要加入jackson 的包。

```xml
<dependency>
    <groupId>com.fasterxml.jackson.core</groupId>
    <artifactId>jackson-databind</artifactId>
    <version>2.9.0</version>
</dependency>
<dependency>
    <groupId>com.fasterxml.jackson.core</groupId>
    <artifactId>jackson-core</artifactId>
    <version>2.9.0</version>
</dependency>
<dependency>
    <groupId>com.fasterxml.jackson.core</groupId>
    <artifactId>jackson-annotations</artifactId>
    <version>2.9.0</version>
</dependency>
```

##### 避免img/css/js被拦截

```xml
DispatcherServlet会拦截到所有资源，导致静态资源（img、css、js）也会被拦截到，从而不能被使用。
解决问题就是需要配置静态资源不进行拦截，在springmvc.xml配置文件添加如下配置
location元素表示webapp目录下的包下的所有文件
mapping元素表示以/static开头的所有请求路径，如/static/a 或者/static/a/b
<!-- 设置静态资源不过滤 -->
<mvc:resources location="/css/" mapping="/css/**"/> <!-- 样式 -->
<mvc:resources location="/images/" mapping="/images/**"/> <!-- 图片 -->
<mvc:resources location="/js/" mapping="/js/**"/> <!-- javascript -->
```

#### 文件上传

##### 文件上传回顾

上传的jar包

```xml
<dependency>
    <groupId>commons-fileupload</groupId>
    <artifactId>commons-fileupload</artifactId>
    <version>1.3.1</version>
</dependency>
<dependency>
    <groupId>commons-io</groupId>
    <artifactId>commons-io</artifactId>
    <version>2.4</version>
</dependency>
```

编写文件上传的JSP页面

```html
<h3>文件上传</h3>
<!--form 表单的 enctype 取值必须是：multipart/form-data
(默认值是:application/x-www-form-urlencoded)
method 属性取值必须是 Post-->
<form action="user/fileupload" method="post" enctype="multipart/form-data">
    选择文件：<input type="file" name="upload"/><br/>
    <input type="submit" value="上传文件"/>
</form>
```

 编写文件上传的Controller控制器

```java
@RequestMapping(value = "/fileupload")
public String fileupload(HttpServletRequest request) throws Exception {
    // 先获取到要上传的文件目录
    String path = request.getSession().getServletContext().getRealPath("/uploads");
    // 创建File对象，一会向该路径下上传文件
    File file = new File(path);
    // 判断路径是否存在，如果不存在，创建该路径
    if (!file.exists()) {
        file.mkdirs();
    }
    // 创建磁盘文件项工厂
    DiskFileItemFactory factory = new DiskFileItemFactory();
    ServletFileUpload fileUpload = new ServletFileUpload(factory);
    // 解析request对象
    List<FileItem> list = fileUpload.parseRequest(request);
    // 遍历
    for (FileItem fileItem : list) {
        // 判断文件项是普通字段，还是上传的文件
        if (fileItem.isFormField()) {
        } else {
            // 上传文件项
            // 获取到上传文件的名称
            String filename = fileItem.getName();
            // 上传文件
            fileItem.write(new File(file, filename));
            // 删除临时文件
            fileItem.delete();
        }
    }
    return "success";
}
```

##### SpringMVC方式上传文件

传统方式的文件上传，指上传文件和访问应用存在于同一台服务器上。且上传完成之后，浏览器可能跳转。

SpringMVC框架提供了MultipartFile对象，该对象表示上传的文件，要求变量名称必须和表单file标签的name属性名称相同。

![1565765043642](https://cdn.jsdelivr.net/gh/siyuanzhou/pic@master/pic/2018-07-07-SpringMVC%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/1565765043642.png)

```java
springMVC上传
<form action="fileUpload2" method="post" enctype="multipart/form-data">
    选择文件：<input type="file" name="upload"/><br/>
    <input type="submit" value="上传文件"/>
</form>
@RequestMapping("/fileUpload2")
public String fileUpload2(HttpServletRequest request, MultipartFile upload) throws IOException {
    System.out.println("fileUpload2执行了。。。");
    String path=request.getSession().getServletContext().getRealPath("/uploads/");
    File file=new File(path);
    if(!file.exists()){
        file.mkdirs();
    }
    String filename=upload.getOriginalFilename();
    String uuid=UUID.randomUUID().toString().replace("-","");
    filename=uuid+"_"+filename;
    //上传文件
    upload.transferTo(new File(file,filename));
    return "success";
}
```

```xml
<!-- 配置配置文件上传解析器 id必须是这个-->
<bean id="multipartResolver" class="org.springframework.web.multipart.commons.CommonsMultipartResolver">
    <!-- 设置上传文件的最大尺寸为 5MB -->
    <property name="maxUploadSize" value="10485760"/>
</bean>
```

##### SpringMVC跨服务器文件上传

![image-20200309224136253](https://cdn.jsdelivr.net/gh/siyuanzhou/pic@master/pic/2018-07-07-SpringMVC%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/image-20200309224136253.png)

首先搭建图片服务器：重新开启一个tomcat模拟图片服务器

其次实现SpringMVC跨服务器方式文件上传，首先导入jar包

```xml
<dependency>
    <groupId>com.sun.jersey</groupId>
    <artifactId>jersey-core</artifactId>
    <version>1.18.1</version>
</dependency>
<dependency>
    <groupId>com.sun.jersey</groupId>
    <artifactId>jersey-client</artifactId>
    <version>1.18.1</version>
</dependency>
```

```java
<h3>跨服务器文件上传</h3>
<form action="user/fileUpload3" method="post" enctype="multipart/form-data">
    选择文件：<input type="file" name="upload"/><br/>
    <input type="submit" value="上传" />
</form>
@RequestMapping("/fileUpload3")
public String fileUpload3(MultipartFile upload) throws IOException {
    //图片服务器上传地址
    String path="http://localhost:9090/fileuploadserver_war_exploded/uploads/";
    String filename=upload.getOriginalFilename();
    String uuid=UUID.randomUUID().toString().replace("-","");
    filename=uuid+"_"+filename;
    // 向图片服务器上传文件
    //创建客户端对象
    Client client=Client.create();
    //连接图片服务器
    WebResource webResource=client.resource(path+filename);
    //上传文件
    webResource.put(upload.getBytes());
    return "success";
}
```

#### SpringMVC的异常处理

Controller调用service，service调用dao，异常都是向上抛出的，最终有DispatcherServlet找异常处理器进
行异常的处理。

![1565770298305](https://cdn.jsdelivr.net/gh/siyuanzhou/pic@master/pic/2018-07-07-SpringMVC%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/1565770298305.png)

##### 自定义异常

```java
//自定义异常类
public class SysException extends Exception {
    private String message;
    @Override
    public String getMessage() {
        return message;
    }
    public void setMessage(String message) {
        this.message = message;
    }
    public SysException(String message) {
        this.message = message;
    }
}
```

```java
//自定义异常处理器
public class SysExceptionResolver implements HandlerExceptionResolver {
    @Override
    public ModelAndView resolveException(HttpServletRequest httpServletRequest, HttpServletResponse httpServletResponse, Object o, Exception ex) {
        ModelAndView mv=new ModelAndView();
        ex.printStackTrace();
        SysException e = null;
        if(ex instanceof  SysException){
            e=(SysException)ex;
        }
        else {
            e=new SysException("请联系管理员解决...");
        }
        mv.addObject("errorMsg",e.getMessage());
        mv.setViewName("error");
        return mv;
    }
}
```

```xml
<!-- 配置异常处理器 -->
<bean id="sysExceptionResolver" class="site.syzhou.exception.SysExceptionResolver"/>
```

#### SpringMVC框架中的拦截器

![image-20200309225758215](https://cdn.jsdelivr.net/gh/siyuanzhou/pic@master/pic/2018-07-07-SpringMVC%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/image-20200309225758215.png)

```java
SpringMVC框架中的拦截器用于对处理器进行预处理和后处理的技术。

可以定义拦截器链，拦接器链就是将拦截器按着一定的顺序结成一条链
在访问被拦截的方法时，拦截器链中的拦截器会按着定义的顺序执行

拦截器也是AOP思想的一种实现方式
想要自定义拦截器，需要实现HandlerInterceptor接口。

HandlerInterceptor接口中的方法
    preHandle方法是controller方法执行前拦截的方法
        可以使用request或者response跳转到指定的页面
        return true放行，执行下一个拦截器，如果没有拦截器，执行controller中的方法。
        return false不放行，不会执行controller中的方法。
        
    postHandle是controller方法执行后执行的方法，在JSP视图执行前。
        可以使用request或者response跳转到指定的页面
		如果指定了跳转的页面，那么controller方法跳转的页面将不会显示。
		
	afterCompletion方法是在JSP执行后执行
		request或者response不能再跳转页面了,在 DispatcherServlet 完全处理完请求后被调用
```

拦截器和过滤器的功能比较类似，有区别

```
1. 过滤器是Servlet规范的一部分，任何框架都可以使用过滤器技术。
2. 拦截器是SpringMVC框架独有的。
3. 过滤器配置了/*，可以拦截任何资源。
4. 拦截器只会对控制器中方法进行拦截。jsp，html,css,image 或者 js 是不会进行拦截的。
```

##### 自定义拦截器

编写拦截器类

```java
public class MyInterceptor1 implements HandlerInterceptor {
    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        System.out.println("preHandle111执行了...");
        return true;
    }
    @Override
    public void postHandle(HttpServletRequest request, HttpServletResponse response, Object handler, ModelAndView modelAndView) throws Exception {
        System.out.println("postHandle111执行了...");
    }
    @Override
    public void afterCompletion(HttpServletRequest request, HttpServletResponse response, Object handler, Exception ex) throws Exception {
        System.out.println("afterCompletion111执行了...");
    }
}
```

```xml
<!-- 配置拦截器 -->
<mvc:interceptors>
    <mvc:interceptor>
        <mvc:mapping path="/user/*"/><!--  用于指定对拦截的 url -->
        <mvc:exclude-mapping path=""/><!--  用于指定排除的 url-->
        <bean class="site.syzhou.intercepter.MyInterceptor1"></bean>
    </mvc:interceptor>
    <mvc:interceptor>
        <mvc:mapping path="/**"/>
        <bean class="site.syzhou.intercepter.MyInterceptor2"></bean>
    </mvc:interceptor>
</mvc:interceptors>
```

![image-20200309230932937](https://cdn.jsdelivr.net/gh/siyuanzhou/pic@master/pic/2018-07-07-SpringMVC%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/image-20200309230932937.png)

##### 用户登录拦截器

```java
public class LoginInterceptor implements HandlerInterceptor{
    @Override
    Public boolean preHandle(HttpServletRequest request,
           HttpServletResponse response, Object handler) throws Exception {
		//如果是登录页面则放行
        if(request.getRequestURI().indexOf("login.action")>=0){
            return true;
        }
        HttpSession session = request.getSession();
        //如果用户已登录也放行
        if(session.getAttribute("user")!=null){
            return true;
        }
        //用户没有登录挑战到登录页面
        request.getRequestDispatcher("/WEB-INF/jsp/login.jsp").forward(request,
                response);
        return false;
    }
}
```

#### SSM搭建整合环境

##### 整合的思路

```
SSM整合可以使用多种方式，选择XML + 注解的方式
先搭建整合的环境
先把Spring的配置搭建完成
再使用Spring整合SpringMVC框架
最后使用Spring整合MyBatis框架
```

![1566445052389](https://cdn.jsdelivr.net/gh/siyuanzhou/pic@master/pic/2018-07-07-SpringMVC%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/1566445052389.png)

##### 項目最终结构

![1566458876928](https://cdn.jsdelivr.net/gh/siyuanzhou/pic@master/pic/2018-07-07-SpringMVC%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/1566458876928.png)



##### 数据库和表结构

```sql
create database ssm;
use ssm;
create table account(
    id int primary key auto_increment,
    name varchar(20),
    money double
);
```

##### pom.xml中依赖配置

```
创建maven的工程（今天会使用到工程的聚合和拆分的概念，这个技术maven高级会讲）
    1. 创建ssm_parent父工程（打包方式选择pom，必须的）
    2. 创建ssm_web子模块（打包方式是war包）
    3. 创建ssm_service子模块（打包方式是jar包）
    4. 创建ssm_dao子模块（打包方式是jar包）
    5. 创建ssm_domain子模块（打包方式是jar包）
    6. web依赖于service，service依赖于dao，dao依赖于domain
    7. 在ssm_parent的pom.xml文件中引入坐标依赖
    8. 部署ssm_web的项目，只要把ssm_web项目加入到tomcat服务器中即可
```

```xml
<properties>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    <maven.compiler.source>1.8</maven.compiler.source>
    <maven.compiler.target>1.8</maven.compiler.target>
    <spring.version>5.0.2.RELEASE</spring.version>
    <slf4j.version>1.6.6</slf4j.version>
    <log4j.version>1.2.12</log4j.version>
    <mysql.version>5.1.6</mysql.version>
    <mybatis.version>3.4.5</mybatis.version>
</properties>
<dependencies>
    <!-- spring -->
    <dependency>
        <groupId>org.aspectj</groupId>
        <artifactId>aspectjweaver</artifactId>
        <version>1.6.8</version>
    </dependency>
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-aop</artifactId>
        <version>${spring.version}</version>
    </dependency>
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-context</artifactId>
        <version>${spring.version}</version>
    </dependency>
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-web</artifactId>
        <version>${spring.version}</version>
    </dependency>
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-webmvc</artifactId>
        <version>${spring.version}</version>
    </dependency>
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-test</artifactId>
        <version>${spring.version}</version>
    </dependency>
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-tx</artifactId>
        <version>${spring.version}</version>
    </dependency>
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-jdbc</artifactId>
        <version>${spring.version}</version>
    </dependency>
    <dependency>
        <groupId>junit</groupId>
        <artifactId>junit</artifactId>
        <version>4.12</version>
        <scope>compile</scope>
    </dependency>
    <dependency>
        <groupId>mysql</groupId>
        <artifactId>mysql-connector-java</artifactId>
        <version>${mysql.version}</version>
    </dependency>
    <dependency>
        <groupId>javax.servlet</groupId>
        <artifactId>servlet-api</artifactId>
        <version>2.5</version>
        <scope>provided</scope>
    </dependency>
    <dependency>
        <groupId>javax.servlet.jsp</groupId>
        <artifactId>jsp-api</artifactId>
        <version>2.0</version>
        <scope>provided</scope>
    </dependency>
    <dependency>
        <groupId>jstl</groupId>
        <artifactId>jstl</artifactId>
        <version>1.2</version>
    </dependency>
    <!-- log start -->
    <dependency>
        <groupId>log4j</groupId>
        <artifactId>log4j</artifactId>
        <version>${log4j.version}</version>
    </dependency>
    <dependency>
        <groupId>org.slf4j</groupId>
        <artifactId>slf4j-api</artifactId>
        <version>${slf4j.version}</version>
    </dependency>
    <dependency>
        <groupId>org.slf4j</groupId>
        <artifactId>slf4j-log4j12</artifactId>
        <version>${slf4j.version}</version>
    </dependency>
    <!-- log end -->
    <dependency>
        <groupId>org.mybatis</groupId>
        <artifactId>mybatis</artifactId>
        <version>${mybatis.version}</version>
    </dependency>
    <dependency>
        <groupId>org.mybatis</groupId>
        <artifactId>mybatis-spring</artifactId>
        <version>1.3.0</version>
    </dependency>
    <dependency>
        <groupId>c3p0</groupId>
        <artifactId>c3p0</artifactId>
        <version>0.9.1.2</version>
        <type>jar</type>
        <scope>compile</scope>
    </dependency>
</dependencies>
```

##### 编写业务层代码

```java
public class Account implements Serializable{
    private static final long serialVersionUID = 7355810572012650248L;
    private Integer id;
    private String name;
    private Double money;
    //生成get和set方法和toString方法
}

@Service("accountService")
public class AccountServiceImpl implements AccountService {
    @Autowired
    private  AccountDao accountDao;
    @Override
    public List<Account> findAll() {
        System.out.println("业务层：查询所有账户...");
        List<Account> accounts=accountDao.findAll();
        return accounts;
    }
    @Override
    public void saveAccount(Account account) {
        System.out.println("业务层：保存账户...");
        accountDao.saveAccount(account);
    }
}

@Repository
public interface AccountDao {
    @Select("select * from account")
    public List<Account> findAll();
    @Insert("insert into account(name,money) value(#{name},#{money})")
    public void saveAccount(Account account);
}
```

##### 创建log4j.properties文件

```properties
# Set root category priority to INFO and its only appender to CONSOLE.
#log4j.rootCategory=INFO, CONSOLE            debug   info   warn error fatal
log4j.rootCategory=info, CONSOLE, LOGFILE

# Set the enterprise logger category to FATAL and its only appender to CONSOLE.
log4j.logger.org.apache.axis.enterprise=FATAL, CONSOLE

# CONSOLE is set to be a ConsoleAppender using a PatternLayout.
log4j.appender.CONSOLE=org.apache.log4j.ConsoleAppender
log4j.appender.CONSOLE.layout=org.apache.log4j.PatternLayout
log4j.appender.CONSOLE.layout.ConversionPattern=%d{ISO8601} %-6r [%15.15t] %-5p %30.30c %x - %m\n

# LOGFILE is set to be a File appender using a PatternLayout.
log4j.appender.LOGFILE=org.apache.log4j.FileAppender
log4j.appender.LOGFILE.File=axis.log
log4j.appender.LOGFILE.Append=true
log4j.appender.LOGFILE.layout=org.apache.log4j.PatternLayout
log4j.appender.LOGFILE.layout.ConversionPattern=%d{ISO8601} %-6r [%15.15t] %-5p %30.30c %x - %m\n
```

#### 搭建测试Spring环境

保证Spring框架在 web 工程中独立运行

##### 编写applicationContext.xml

编写spring 配置文件applicationContext.xml并导入约束

在ssm_web项目中的resources目录创建applicationContext.xml的配置文件，编写具体的配置信息，为了节省空间，下面是所有整合后的完整配置文件。

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:context="http://www.springframework.org/schema/context"
       xmlns:aop="http://www.springframework.org/schema/aop"
       xmlns:tx="http://www.springframework.org/schema/tx"
       xsi:schemaLocation="http://www.springframework.org/schema/beans
                    http://www.springframework.org/schema/beans/spring-beans.xsd
                    http://www.springframework.org/schema/context
                    http://www.springframework.org/schema/context/spring-context.xsd
                    http://www.springframework.org/schema/aop
                    http://www.springframework.org/schema/aop/spring-aop.xsd
                    http://www.springframework.org/schema/tx
                    http://www.springframework.org/schema/tx/spring-tx.xsd">
    <!-- 开启注解扫描，要扫描的是service和dao层的注解，要忽略web层controller注解，因为web层让SpringMVC框架
    去管理 -->
    <context:component-scan base-package="site.newvalue">
        <!-- 配置要忽略的注解 -->
        <context:exclude-filter type="annotation"
                                expression="org.springframework.stereotype.Controller"/>
    </context:component-scan>
    <!-- spring整合Mybatis框架，把mybatis配置文件(SqlMapConfig.xml)中内容配置到spring配置文件中-->
    <!-- 配置C3P0的连接池对象 -->
    <bean id="datasource" class="com.mchange.v2.c3p0.ComboPooledDataSource">
        <property name="driverClass" value="com.mysql.jdbc.Driver" />
        <property name="jdbcUrl" value="jdbc:mysql:///ssm" />
        <property name="user" value="root" />
        <property name="password" value="123456" />
    </bean>
    <!-- 配置SqlSession的工厂 -->
    <bean id="sqlSessionFactoryBean" class="org.mybatis.spring.SqlSessionFactoryBean">
        <!-- 数据库连接池 -->
        <property name="dataSource" ref="datasource"></property>
    </bean>
    <!-- 配置扫描dao的包,配置自动扫描所有Mapper接口和文件 -->
    <!-- 配置扫描dao的包 -->
    <bean id="mapperScanner" class="org.mybatis.spring.mapper.MapperScannerConfigurer">
        <property name="basePackage" value="site.newvalue.dao"/>
    </bean>
    <!--配置 spring 声明式事务管理-->
    <!-- 配置事务管理器 -->
    <bean class="org.springframework.jdbc.datasource.DataSourceTransactionManager" id="dataSourceTransactionManager">
        <property name="dataSource" ref="datasource"></property>
    </bean>
    <!-- 配置事务的通知 -->
    <tx:advice transaction-manager="dataSourceTransactionManager" id="txAdvice">
        <tx:attributes>
            <tx:method name="*" read-only="false" isolation="DEFAULT"/>
            <tx:method name="find*" read-only="true"></tx:method>
        </tx:attributes>
    </tx:advice>
    <!--配置AOP-->
    <aop:config>
        <!-- 配置切入点表达式 -->
        <aop:pointcut id="pt1" expression="execution(* site.newvalue.service.impl.*.*(..))"/>
        <!-- 建立通知和切入点表达式的关系 -->
        <aop:advisor advice-ref="txAdvice" pointcut-ref="pt1"></aop:advisor>
    </aop:config>
</beans>
```

##### 注解配置业务层代码

见上面业务层代码service和dao中的@Service("accountService")，@Repository等

##### 测试Spring

只用于测试，对整合用处不大，可以看到，测试类使用spring注解得到了accountService类

```java
public class TestSpring {
    @Test
    public void runTestSpring(){
        //加载配置文件
        ApplicationContext ac=new ClassPathXmlApplicationContext("classpath:applicationContext.xml");
        //获取对象
        AccountService as = (AccountService) ac.getBean("accountService");
        //调用方法
        as.findAll();
    }
}
```

#### 整合SpringMVC环境

目的：在controller中能成功的调用service对象中的方法

原理：在web.xml中配置ContextLoaderListener监听器（该监听器只能加载WEB-INF目录下的applicationContext.xml的配置文件，所以需要配置applicationContext.xml的位置）。

这样在项目启动的时候，就去加载applicationContext.xml的配置文件。监听器原理如下图：

![1566445076808](https://cdn.jsdelivr.net/gh/siyuanzhou/pic@master/pic/2018-07-07-SpringMVC%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/1566445076808.png)

##### 修改web.xml文件

```
1. 在web.xml中配置DispatcherServlet前端控制器
2. 在web.xml中配置DispatcherServlet过滤器解决中文乱码
```

下面是整合后的web.xml文件，监听器是用于整合springmvc的，这三个是web.xml三大组件

```xml
<!DOCTYPE web-app PUBLIC
 "-//Sun Microsystems, Inc.//DTD Web Application 2.3//EN"
 "http://java.sun.com/dtd/web-app_2_3.dtd" >
<web-app>
  <display-name>Archetype Created Web Application</display-name>
  
  <!-- 配置前端控制器：服务器启动必须加载，需要加载springmvc.xml配置文件 -->
  <servlet>
    <servlet-name>dispatcherServlet</servlet-name>
    <servlet-class>org.springframework.web.servlet.DispatcherServlet</servlet-class>
    <!-- 配置初始化参数，创建完DispatcherServlet对象，加载springmvc.xml配置文件 -->
    <init-param>
      <param-name>contextConfigLocation</param-name>
      <param-value>classpath:springmvc.xml</param-value>
    </init-param>
    <!-- 服务器启动的时候，让DispatcherServlet对象创建 -->
    <load-on-startup>1</load-on-startup>
  </servlet>
  <servlet-mapping>
    <servlet-name>dispatcherServlet</servlet-name>
    <url-pattern>/</url-pattern>
  </servlet-mapping>
  
  <!-- 配置解决中文乱码的过滤器 -->
  <filter>
    <filter-name>characterEncodingFilter</filter-name>
    <filter-class>org.springframework.web.filter.CharacterEncodingFilter</filter-class>
    <init-param>
      <param-name>encoding</param-name>
      <param-value>UTF-8</param-value>
    </init-param>
  </filter>
  <filter-mapping>
    <filter-name>characterEncodingFilter</filter-name>
    <url-pattern>/*</url-pattern>
  </filter-mapping>
  
  <!-- 配置Spring的监听器,在项目启动的时候，就去加载applicationContext.xml的配置文件,默认只加载web-inf目录下的applicationContext.xml -->
  <listener>
    <listener-class>org.springframework.web.context.ContextLoaderListener</listener-class>
  </listener>
  <!--设置applicationContext.xml配置文件位置-->
  <context-param>
    <param-name>contextConfigLocation</param-name>
    <param-value>classpath:applicationContext.xml</param-value>
  </context-param>
</web-app>
```

##### 创建编写springmvc.xml

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:mvc="http://www.springframework.org/schema/mvc"
       xmlns:context="http://www.springframework.org/schema/context"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="
        http://www.springframework.org/schema/beans
        http://www.springframework.org/schema/beans/spring-beans.xsd
        http://www.springframework.org/schema/mvc
        http://www.springframework.org/schema/mvc/spring-mvc.xsd
        http://www.springframework.org/schema/context
        http://www.springframework.org/schema/context/spring-context.xsd">
    <!-- 扫描controller的注解，别的不扫描 -->
    <context:component-scan base-package="site.syzhou">
        <context:include-filter type="annotation" expression="org.springframework.stereotype.Controller"/>
    </context:component-scan>
    <!-- 配置视图解析器 -->
    <bean id="internalResourceViewResolver" class="org.springframework.web.servlet.view.InternalResourceViewResolver">
        <property name="prefix" value="/WEB-INF/pages/"></property>
        <property name="suffix" value=".jsp"></property>
    </bean>
    <!-- 设置静态资源不过滤 -->
    <mvc:resources location="/css/" mapping="/css/**" />
    <mvc:resources location="/images/" mapping="/images/**" />
    <mvc:resources location="/js/" mapping="/js/**" />
    <!-- 开启对SpringMVC注解的支持 -->
    <mvc:annotation-driven/>
</beans>
```

##### 测试SpringMVC

1 编写前端页面

```jsp
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
<head>
    <title>index</title>
</head>
<body>
    <a href="account/findAll">查询所有</a><br>
    <h3>测试保存</h3>
    <form method="post" action="account/save">
        姓名：<input type="text" name="name"><br>
        金额：<input type="text" name="money"><br>
        <input type="submit" value="保存">
    </form>
</body>
</html>

//list.jsp
<%@ page contentType="text/html;charset=UTF-8" language="java" isELIgnored="false" %>
<%@taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<html>
<head>
    <title>Title</title>
</head>
<body>
    <b>查询了所有的账户信息</b><br>
    <c:forEach items="${list}" var="account">
        ${account.name}<br>
    </c:forEach>
</body>
</html>
```

2 创建AccountController类，编写方法，进行测试

```java
@Controller
@RequestMapping("/account")
public class AccountController {
    //利用spring自动注入对象
    @Autowired
    private AccountService accountService;
    @RequestMapping("/findAll")
    public String findAll(Model model){
        System.out.println("表现层：查询所有账户信息");
        //调用service方法
        List<Account> list=accountService.findAll();
        model.addAttribute("list",list);
        return "list";
    }
    @RequestMapping("/save")
    public void save(Account account, HttpServletRequest request, HttpServletResponse response) throws IOException {
        System.out.println("表现层：保存账户信息");
        //调用service方法
        accountService.saveAccount(account);
        response.sendRedirect(request.getContextPath()+"/account/findAll");
        return ;
    }
}
```

#### 整合MyBatis的环境

目的：把SqlMapConfig.xml配置文件中的内容配置到applicationContext.xml配置文件中

##### 编写SqlMapConfig.xml配置文件

只用于测试mybatis是否成功，后面整合时用的是前面整合思路applicationContext.xml，是cp30数据源

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE configuration
        PUBLIC "-//mybatis.org//DTD Config 3.0//EN"
        "http://mybatis.org/dtd/mybatis-3-config.dtd">
<configuration>
    <!--使用typeAliases配置别名，它只能配置domain中类的别名 -->
    <typeAliases>
        <!--<typeAlias type="com.itheima.domain.User" alias="user"></typeAlias>-->
        <package name="site.newvalue.domain"/>
    </typeAliases>
    <!--配置环境-->
    <environments default="mysql">
        <environment id="mysql">
            <transactionManager type="JDBC"/>
            <dataSource type="POOLED">
                <property name="driver" value="com.mysql.jdbc.Driver"/>
                <property name="url" value="jdbc:mysql:///ssm"/>
                <property name="username" value="root"/>
                <property name="password" value="123456"/>
            </dataSource>
        </environment>
    </environments>
    <!-- 使用的是注解，引入映射配置文件 -->
    <mappers>
        <!-- <mapper class="site.syzhou.dao.AccountDao"/> -->
        <!-- <mapper resources="site/syzhou/dao/AccountDao.xml"/> -->
        <!-- 默认扫描该包下所有的dao接口都可以使用 -->
        <package name="site.newvalue.dao"/>
    </mappers>
</configuration>
```

##### Dao接口方法上加注解

见上面的实现dao接口

##### 测试MyBatis

只用于测试，对整合无帮助

```java
public class TestMyBatis {
    @Test
    public void runTestMybatis() throws IOException {
        InputStream in= Resources.getResourceAsStream("SqlMapConfig.xml");
        //2.创建SqlSessionFactory工厂
        SqlSessionFactoryBuilder builder=new SqlSessionFactoryBuilder();
        SqlSessionFactory factory=builder.build(in);
        //3.使用工厂生产SqlSession对象
        SqlSession sqlSession=factory.openSession();
        //4.使用SqlSession创建Dao接口的代理对象
        AccountDao accountDao=sqlSession.getMapper(AccountDao.class);
        accountDao.saveAccount(new Account("4",4.0));
        sqlSession.commit();
        List<Account> accounts=accountDao.findAll();
        for(Account account : accounts){
            System.out.println(account);
        }
        sqlSession.close();
    }
}
```

#### 整合SpringMVC详细

目的：在controller中能成功的调用service对象中的方法

原理：在web.xml中配置ContextLoaderListener监听器（该监听器只能加载WEB-INF目录下的applicationContext.xml的配置文件，所以需要配置applicationContext.xml的位置）。

这样在项目启动的时候，就去加载applicationContext.xml的配置文件。监听器原理如下图：

![1566445076808](https://cdn.jsdelivr.net/gh/siyuanzhou/pic@master/pic/2018-07-07-SpringMVC%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/1566445076808.png)

```xml
<!-- 在web.xml中配置Spring的监听器,在项目启动的时候，就去加载applicationContext.xml的配置文件,默认只加载web-inf目录下的applicationContext.xml -->
<listener>
    <listener-class>org.springframework.web.context.ContextLoaderListener</listener-class>
</listener>
<!--设置applicationContext.xml配置文件位置-->
<context-param>
    <param-name>contextConfigLocation</param-name>
    <param-value>classpath:applicationContext.xml</param-value>
</context-param>
```

接着在controller中注入service对象，调用service对象的方法进行测试

```java
//利用spring自动注入对象
@Autowired
private AccountService accountService;
```

#### 整合MyBatis详细

目的：把SqlMapConfig.xml配置文件中的内容配置到applicationContext.xml配置文件中

```xml
<!-- spring整合Mybatis框架，把mybatis配置文件(SqlMapConfig.xml)中内容配置到spring配置文件中-->
<!-- 配置C3P0的连接池对象 -->
<bean id="datasource" class="com.mchange.v2.c3p0.ComboPooledDataSource">
    <property name="driverClass" value="com.mysql.jdbc.Driver" />
    <property name="jdbcUrl" value="jdbc:mysql:///ssm" />
    <property name="user" value="root" />
    <property name="password" value="root" />
</bean>
<!-- 配置SqlSession的工厂 -->
<bean id="sqlSessionFactoryBean" class="org.mybatis.spring.SqlSessionFactoryBean">
    <!-- 数据库连接池 -->
    <property name="dataSource" ref="datasource"></property>
</bean>
<!-- 配置扫描dao的包,配置自动扫描所有Mapper接口和文件 -->
<!-- 配置扫描dao的包 -->
<bean id="mapperScanner" class="org.mybatis.spring.mapper.MapperScannerConfigurer">
    <property name="basePackage" value="site.syzhou.dao"/>
</bean>
```

2.在AccountDao接口中添加@Repository注解

3.在service中注入dao对象，进行测试

```java
@Autowired
private  AccountDao accountDao;
```

4. 配置Spring的声明式事务管理

```xml
<!--配置 spring 声明式事务管理-->
<!-- 配置事务管理器 -->
<bean class="org.springframework.jdbc.datasource.DataSourceTransactionManager" id="dataSourceTransactionManager">
    <property name="dataSource" ref="datasource"></property>
</bean>
<!-- 配置事务的通知 -->
<tx:advice transaction-manager="dataSourceTransactionManager" id="txAdvice">
    <tx:attributes>
        <tx:method name="*" read-only="false" isolation="DEFAULT"/>
        <tx:method name="find*" read-only="true"></tx:method>
    </tx:attributes>
</tx:advice>
<!--配置AOP-->
<aop:config>
    <!-- 配置切入点表达式 -->
    <aop:pointcut id="pt1" expression="execution(* site.syzhou.service.impl.*.*(..))"/>
    <!-- 建立通知和切入点表达式的关系 -->
    <aop:advisor advice-ref="txAdvice" pointcut-ref="pt1"></aop:advisor>
</aop:config>
```

