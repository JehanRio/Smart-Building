#pragma once
#include<mysql.h>
#include<chrono>	// ʱ����
#include<iostream>
using namespace std;
using namespace chrono;
class MysqlConn
{
public:
	// ��ʼ�����ݿ�����
	MysqlConn();
	// �ͷ����ݿ�����
	~MysqlConn();
	// �������ݿ�
	bool connect(string user, string passwd, string dbName, string ip, unsigned short port = 3306);
	// �������ݿ⣺ insert,update,delete
	bool update(string sql);
	// ��ѯ���ݿ�
	bool query(string sql);
	// ������ѯ�õ��Ľ����
	bool next();
	// �õ�������е��ֶ�ֵ
	string Value(int index);
	// �������
	bool transaction();
	// �ύ����
	bool commit();
	// �������
	bool rollback();
	// ˢ����ʼ�Ŀ���ʱ���
	void refreshAliveTime();
	// �������Ӵ������ʱ��
	long long getAliveTime();
private:
	void freeResult();
	MYSQL* m_conn = nullptr;
	MYSQL_RES* m_result = nullptr;
	MYSQL_ROW m_row = nullptr;	// ����ָ�� ���ڱ����ȡ�Ľ��
	steady_clock::time_point m_alivetime;
};
