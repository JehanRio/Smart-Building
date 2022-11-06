#include "MysqlConn.h"

MysqlConn::MysqlConn()
{
	m_conn = mysql_init(nullptr);
	mysql_set_character_set(m_conn, "utf8");  // ���ñ���
}

MysqlConn::~MysqlConn()
{
	if (m_conn != nullptr)
	{
		mysql_close(m_conn);
	}
	freeResult();
}

bool MysqlConn::connect(string user, string passwd, string dbName, string ip, unsigned short port)
{
	MYSQL* ptr = mysql_real_connect(m_conn, ip.c_str(), user.c_str(), passwd.c_str(), dbName.c_str(), port, nullptr, 0);	// ת����char*����
	return ptr != nullptr;	// ���Ϊ�գ���Ϊfalse
}

bool MysqlConn::update(string sql)
{
	if (mysql_query(m_conn, sql.c_str()))	// ����0��ִ�гɹ�
	{
		return false;
	}
	return true;
}

bool MysqlConn::query(string sql)
{
	freeResult();
	if (mysql_query(m_conn, sql.c_str()))
	{
		return false; 
	}
	// ʹ��mysql_query������SQL��䣬Ȼ��ʹ��mysql_store_result��mysql_use_result����ȡ����
	m_result = mysql_store_result(m_conn);	// ��mysql�ͻ��˵õ������
	return true;
}

bool MysqlConn::next()
{
	if (m_result != nullptr)
	{
		m_row = mysql_fetch_row(m_result);
		if (m_row != nullptr)
			return true;
	}
	return false;
}

string MysqlConn::Value(int index)
{
	int columnCount = mysql_num_fields(m_result);
	if (index >= columnCount || index < 0)
	{
		return string();
	}
	char* val = m_row[index];
	unsigned long length = mysql_fetch_lengths(m_result)[index];
	return string(val,length);
}

bool MysqlConn::transaction()
{
	return mysql_autocommit(m_conn,false);	// �ֶ��ύ
}

bool MysqlConn::commit()
{
	return mysql_commit(m_conn);
}

bool MysqlConn::rollback()
{
	return mysql_rollback(m_conn);
}

void MysqlConn::refreshAliveTime()
{
	m_alivetime = steady_clock::now();
}

long long MysqlConn::getAliveTime()
{
	nanoseconds res = steady_clock::now() - m_alivetime;	// �߾���:����
	milliseconds millsec = duration_cast<milliseconds>(res);	// �߾���->�;��ȣ�����
	return millsec.count();	// �õ���ʱ���������ж��ٸ�����
}

void MysqlConn::freeResult()
{
	if (m_result)
	{
		mysql_free_result(m_result);
		m_result = nullptr;
	}
}
