''' 1965. Employees With Missing Information ''' 
with name_null as (
    select e.employee_id , s.salary , e.name 
    from employees e 
    left join salaries s 
    on e.employee_id = s.employee_id
),
salary_null as (
    select s.employee_id, s.salary, e.name
    from salaries s
    left join employees e
    on e.employee_id = s.employee_id 
),
temp_temp as (
    select employee_id  
    from name_null 
    where salary is null 

    union all 

    select employee_id 
    from salary_null 
    where name is null  
) 
select employee_id 
from temp_temp
order by employee_id ;

'''
1978. Employees Whose Manager Left the Company
  '''

with conditional_employees as (
    select employee_id , manager_id
    from employees 
    where salary < 30000
), 
temp_temp as (
    select employee_id 
    from conditional_employees 
    where manager_id not  in (
        select employee_id 
        from employees
    ) 
     
)
select employee_id 
from temp_temp 
order by employee_id ; 


'''
  2356. Number of Unique Subjects Taught by Each Teacher
''' 
select teacher_id , 
    count(distinct subject_id) as cnt 
from teacher 
group by teacher_id ; 



