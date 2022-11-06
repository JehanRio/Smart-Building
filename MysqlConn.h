#pragma once
#include<mysql.h>
#include<chrono>	// 时钟类
#include<iostream>
using namespace std;
using namespace chrono;
class MysqlConn
{
public:
	// 初始化数据库连接
	MysqlConn();
	// 释放数据库连接
	~MysqlConn();
	// 连接数据库
	bool connect(string user, string passwd, string dbName, string ip, unsigned short port = 3306);
	// 更新数据库： insert,update,delete
	bool update(string sql);
	// 查询数据库
	bool query(string sql);
	// 遍历查询得到的结果集
	bool next();
	// 得到结果集中的字段值
	string Value(int index);
	// 事务操作
	bool transaction();
	// 提交事务
	bool commit();
	// 事务滚回
	bool rollback();
	// 刷新起始的空闲时间点
	void refreshAliveTime();
	// 计算连接存货的总时长
	long long getAliveTime();
private:
	void freeResult();
	MYSQL* m_conn = nullptr;
	MYSQL_RES* m_result = nullptr;
	MYSQL_ROW m_row = nullptr;	// 二级指针 用于保存获取的结果
	steady_clock::time_point m_alivetime;
};

