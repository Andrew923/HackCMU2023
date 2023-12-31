CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230706000000_e20230706235959_p20230707021740_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-07-07T02:17:40.805Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-07-06T00:00:00.000Z   time_coverage_end         2023-07-06T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx���   
�          A2{�{@��?O\)@���C	{�{@�  ?��
@�(�C	T{                                    Bx���  N          A1�� ��@�(�>�G�@�\C�=� ��@��H?n{@���C�3                                    Bx��L  �          A1���
=@�=�?��C
��
=@���?��@FffC.                                    Bx��%�  �          A0����@�{>\)?:�HCٚ��@�p�?�R@N�RC�                                    Bx��4�  �          A1G�� ��@�33>Ǯ@ ��C�H� ��@��?\(�@�
=C�                                    Bx��C>  �          A2ff���@���>�(�@p�C�����@�?fff@���C�                                    Bx��Q�  �          A2ff�p�@��?   @%C���p�@�?z�H@���C�q                                    Bx��`�  �          A2{�p�@�(�>��H@#33C���p�@��H?u@�
=C�
                                    Bx��o0  T          A1p�� ��@��
>�@
=C�=� ��@�\?k�@���C�3                                    Bx��}�  
�          A1����@���?
=@B�\C
G����@�33?��@��
C
z�                                    Bx���|  �          A2{��@�(�>.{?\(�C
u���@�33?�R@K�C
��                                    Bx���"  "          A1G�� z�@�>�?333C��� z�@�\?��@EC�H                                    Bx����  "          A1G�����@��H?��\@���C�q����@�Q�?��R@�  C                                    Bx���n  N          A1����  @���?n{@�G�Cs3��  @��\?�33@�C�3                                    Bx���  �          A1G����\@���?aG�@���C)���\@�\)?��@�ffCY�                                    Bx��պ  �          A1���G�@�=q?^�R@��C����G�@�  ?�=q@���C&f                                    Bx���`  �          A1G����@�Q�?Tz�@���C^����@�ff?��@�C�)                                    Bx���  �          A1���
=@��R?E�@~�RC���
=@��?�(�@�=qC(�                                    Bx���  �          A1���@��?Y��@�(�C����@�?�ff@�
=C��                                    Bx��R  �          A0������@�ff?Y��@�p�C������@�(�?�ff@�  C�R                                    Bx���  �          A1�   @���?n{@��CG��   @�\?�\)@�33C��                                    Bx��-�  �          A0z���\)@��?aG�@��\C���\)@�  ?���@�33C                                    Bx��<D  �          A0  ��ff@�
=?�z�@�z�C�=��ff@�(�?�A{C	&f                                    Bx��J�  �          A,�����
@�?���A
=C�����
@�=q@\)A?\)C.                                    Bx��Y�  �          A-G����@��?
=@H��CT{���@�{?��
@��RC�                                     Bx��h6  �          A-����\@�p�?�=q@ᙚC�����\@�\?�  AQ�C�f                                    Bx��v�  �          A.{��Q�@���?�@��Cٚ��Q�@�ff?˅A=qC#�                                    Bx����  T          A/\)� Q�@�G�?�33@�C	�R� Q�@�ff?�ffA  C
{                                    Bx���(  T          A)���@�?��@�=qC� ��@��H?�  A�C
                                    Bx����  
�          A$����\)@�ff?��@�  C���\)@��
?��
A�Cs3                                    Bx���t  
Z          A#33��@��?���@���CxR��@�\)?޸RAp�C�
                                    Bx���  
�          Aff�ᙚ@�z�?u@��HC���ᙚ@ڏ\?��@�Q�C�f                                    Bx����  T          A(��ҏ\@�?   @<(�C��ҏ\@�z�?c�
@�
=C��                                    Bx���f  �          A=q��\)@�{?��@]p�C���\)@���?z�H@��RCJ=                                    Bx���  T          A"�R��ff@ڏ\?��@�ffC	� ��ff@�Q�?�G�A(�C	�=                                    Bx����  
�          A z���33@�  ?��@��HCaH��33@�{?�Q�A�C��                                    Bx��	X  T          A �����@�(�?��@�{C�\���@��?�Ap�C�{                                    Bx���  
�          A"{���
@�z�?}p�@��HC�����
@�\?�{@�
=C#�                                    Bx��&�  �          A"=q���H@�ff?�R@_\)C�����H@��?�  @�{C��                                    Bx��5J  T          A!�����
@���<�>8Q�C@ ���
@�G�>�G�@�RCJ=                                    Bx��C�  �          A&�R���A�<��
=��
C����A ��>�(�@�C(�                                    Bx��R�  �          A)���  A33���\)Cu���  A
=>Ǯ@�C}q                                    Bx��a<  
�          A(  ����A Q�=�G�?
=C:�����A   ?�\@3�
CJ=                                    Bx��o�  �          A+
=��z�Aff>#�
?W
=C33��z�A=q?�@B�\CE                                    Bx��~�  �          A(����
=A=q>.{?h��C�=��
=A�?�@G
=C��                                    Bx���.  T          A)������A ��>�(�@�
C�f����A (�?Q�@��RC                                    Bx����  "          A,����33A�R>��@   C����33A{?^�R@�(�C�                                    Bx���z  
�          A-G���\)AQ�>�33?���C&f��\)A�
?@  @~�RC=q                                    Bx���   �          A*ff��Q�A\)>�\)?\Cs3��Q�A
=?.{@j=qC��                                    Bx����  �          A*�H���
A�R>��R?�z�C\���
Aff?333@p��C#�                                    Bx���l  T          A,Q���(�Az�>L��?��C�q��(�A(�?
=@H��C�\                                    Bx���  "          A,z���33AG�>�?+�Ch���33A�?�@1G�Cu�                                    Bx���  "          A,�����A	G����
��Q�B�Ǯ���A	�>Ǯ@33B��
                                    Bx��^  "          A-p��љ�A
{<��
=�B�ff�љ�A	�>�
=@  B�u�                                    Bx��  T          A-p�����A��>.{?h��B�����A��?�@@  B��)                                    Bx���  �          A,(�����A�>�z�?��C &f����A33?.{@eC 8R                                    Bx��.P  T          A0����ffA\)��\)��p�B�z���ffA\)>���?޸RB��                                    Bx��<�  T          A1G���(�A���������B�B���(�A�=�G�?�B�=q                                    Bx��K�  �          A2�R����A�þ�33��B��f����A�=u>��
B��H                                    Bx��ZB  "          A0����
=A=q��
=��B�p���
=Aff���
��Q�B�aH                                    Bx��h�  T          A/���{A�;��R����B���{A��=�Q�>��B��R                                    Bx��w�  T          A*=q���RA�;��R��Q�B�.���RA��=�\)>�Q�B�(�                                    Bx���4  
�          A#����A33�0���w�B������A����
��\B�                                    Bx����  �          A"�\��z�A�
�.{�tz�B����z�A(����R��(�B���                                    Bx����  �          A#���  A  ����XQ�B����  AQ�k����B��
                                    Bx���&  �          A!����ffA
{�5��  B����ffA
ff��33���HB���                                    Bx����  �          A Q���ffA
=q���\��33B����ffA
�H�(���s33B�u�                                    Bx���r  �          A!���\)A�Ϳ�����G�B�=��\)Ap��u��\)B�L�                                    Bx���  �          A!�����
A\)�У����B�
=���
AQ쿢�\��
=B�q                                    Bx���  "          A"=q���
AzῸQ��=qB�����
Ap������\B�aH                                    Bx���d  T          A#33�\)A(���p��B���\)A���\)����B�k�                                    Bx��

  T          A#\)����A(���(��  B�.����A�ÿ�����ffB���                                    Bx���  "          A!���^{A�׿���,��B��^{A�����
��Bܽq                                    Bx��'V  �          A$z��a�AG��p��]G�B�ff�a�A�R�ff�<z�B�
=                                    Bx��5�  T          A!p�����@�ff�L�;��B�� ����@�ff>�  ?��HB��=                                    Bx��D�  T          A ����33@�G�?
=q@E�C0���33@�Q�?Q�@�CJ=                                    Bx��SH  �          A$�����HA	녿�\)��  B�����HA
ff�O\)��\)B��R                                    Bx��a�  (          A.�H��  A�׿�z��\B�#���  A��Tz����HB��                                    Bx��p�  �          A2ff����A��>�ff@z�B��H����AG�?E�@}p�B�                                    Bx��:  
�          A2=q���Ap�@{A8Q�C����A (�@ ��APz�Cff                                    Bx����  �          A3��ƸRA�
������p�B����ƸRAzῃ�
��  B��{                                    Bx����  T          A3
=��=qA�H�h����{B�\��=qA\)����C�
B��                                    Bx���,  �          A4  ��G�A=q<��
=�Q�B�(���G�A=q>���?�
=B�33                                    Bx����  T          A4Q���(�A?c�
@���B����(�AG�?�
=@�G�B��f                                    Bx���x  T          A4Q���G�A  >�ff@33B�k���G�A�?B�\@w�B��=                                    Bx���  �          A3���ffA�R?��\@��
B�����ffA=q?��@��B��)                                    Bx����  T          A2=q��\)@���@@  Az�\CE��\)@陚@O\)A��C��                                    Bx���j  �          A2ff���@�\)@Q�AE�C�q���@���@'�AYC
=                                    Bx��  �          A2�H�ڏ\A
ff?�AG�C T{�ڏ\A	G�@�A+�
C �                                    Bx���  �          A5G���ffAff?�  @�{B�����ffA��?��
A��B�L�                                    Bx�� \  T          A7
=��33A%녿�(���B���33A&�\��z����
B�W
                                    Bx��/  �          A6ff���A%��G��|(�B�\)���A%p�������B�G�                                    Bx��=�  �          A4  ��\)A�
?��@�33B���\)A33?Ǯ@��B�B�                                    Bx��LN  
�          A333����A��?�  @��B�8R����A��?�G�@�Q�B�u�                                    Bx��Z�  �          A3
=���A{?��
A�HB��
���AG�@�A'�B�.                                    Bx��i�  �          A2�R���A��?��HA��B��
���A�
?��HA!��B�#�                                    Bx��x@  �          A3���\)A�\?�ffA�
B�k���\)A@33A(z�B��R                                    Bx����  �          A3�
��z�A��?�=qA��B����z�A�
?�=qA=qB�.                                    Bx����  �          A3�
�ÅA��?��A=qB����ÅA�
?��AffB��f                                    Bx���2  T          A3���
=A�R?��
A��B�B���
=A�@ ��A%G�B��\                                    Bx����  �          A3�
��AQ�@^�RA���B�\)��A�H@mp�A�z�B��f                                    Bx���~  T          A5��{A�?�33AffB�(���{AQ�@Q�A-G�B�p�                                    Bx���$  T          A733��p�Aff@�A+
=B��\��p�A��@ffA=B��)                                    Bx����  �          A733��{A�R@G�A"ffB��{��{A�@\)A4��B��)                                    Bx���p  T          A5����
A�
@�A1�B�u����
A
=@=qAC\)B�                                    Bx���  �          A6{��G�A�@ ��A#33B�G���G�Az�@�RA5�B�=                                    Bx��
�  �          A6�\��33A��?˅A z�B���33AQ�?�A=qB�3                                    Bx��b  �          A5���z�A�R?�
=A(�B�8R��z�A{?��A��B�p�                                    Bx��(  �          A6�\����A
=?�
=@��RB�����A�\?��@�  B�Ǯ                                    Bx��6�  �          A6�R�ƸRA\)?�  @�G�B����ƸRA�H?�Q�Az�B�Ǯ                                    Bx��ET  T          A6�R��p�A  ?�{@ۅB�����p�A�?Ǯ@��HB�#�                                    Bx��S�  �          A6�\��ffA   ?�@&ffB�8R��ffA�?5@eB�G�                                    Bx��b�  �          A5p���Q�A?��@���B�aH��Q�AG�?��
@�\)B�                                     Bx��qF  �          A6=q���
A��?��@���B�3���
A(�?�(�@�RB���                                    Bx���  �          A5���
=Aff>�
=@��B��{��
=A=q?
=@>�RB���                                    Bx����  T          A5���HA?:�H@mp�B�33���HAp�?fff@���B�G�                                    Bx���8  �          A6=q��33A&�\�#�
�aG�B���33A&�\>�?.{B�                                    Bx����  �          A6�H���
A?�@+�B�����
A��?0��@^�RB���                                    Bx����  "          A6�R����A"�\>#�
?L��B�������A"ff>��
?�{B���                                    Bx���*  �          A6{�p��A*�H��R�H��B�Ǯ�p��A+
=����
B�                                    Bx����  �          A6ff���RA'�
���;�B����RA'�
=u>���B�                                    Bx���v  �          A6=q���A%�<��
=��
B������A%�>#�
?Q�B���                                    Bx���  �          A6�\����A�>��?��B������A�>���@   B��                                    Bx���  �          A5���Q�A(�����G�B�8R��Q�A(����#�
B�33                                    Bx��h  "          A7\)����A��?�R@G
=C
����A��?:�H@k�C�                                    Bx��!  �          A6�R����A�\?aG�@��CE����Aff?z�H@�CO\                                    Bx��/�  �          A6{���A   ?�ff@�G�C�q���@�\)?��@�Q�C�                                    Bx��>Z  
�          A6�\� Q�@��?�\)@�(�C�� Q�@�
=?���@�\C+�                                    Bx��M   �          A6{��\)A Q�?�(�@�(�C�{��\)A (�?�ff@��C�H                                    Bx��[�  �          A4���@�Q�?���@�{CE�@�  ?��@�\CQ�                                    Bx��jL  �          A4  � Q�@��?�G�@��RC�� Q�@�
=?�=qAG�C
                                    Bx��x�  �          A4����z�A�
?^�R@�{C���z�A�
?s33@�=qC�                                    Bx����  �          A4(���\)AG�>�  ?��\C Q���\)AG�>��
?��C T{                                    Bx���>  �          A5���A��>�
=@��C:���A��>��H@�RC=q                                    Bx����  T          A5���ҏ\A�=�?�RB���ҏ\A�>B�\?s33B��                                    Bx����  �          A5G��ȣ�A�R���R����B�B��ȣ�A�R��  ��G�B�B�                                    Bx���0  �          A4�����
AG����.{B������
AG�<��
=�B���                                    Bx����  �          A4����\)A�=�\)>�p�B�.��\)A�=�?�RB�.                                    Bx���|  T          A2�\���
A�>��@ffB�\���
A�>�ff@�B�{                                    Bx���"  T          A2�H���AG�<�>#�
B��)���AG�=�\)>�Q�B��)                                    Bx����  �          A/
=�׮A	G�?
=q@5C .�׮A	�?�@@  C .                                    Bx��n  T          A1G�����A(�<��
=��
B������A(�=#�
>k�B��                                    Bx��  �          A/���A��>�
=@��B��q��A��>�G�@z�B��q                                    Bx��(�  T          A0Q���33A�H=�\)>�Q�B�.��33A�H=�Q�>�ffB�.                                    Bx��7`  �          A/�
��p�A�\>.{?^�RB����p�A�\>8Q�?n{B��                                    Bx��F  T          A2�H��
=A��>��@ffB���
=A��>�
=@Q�B�                                    Bx��T�  �          A3���33A�>Ǯ?�p�B��
��33A�>Ǯ?�p�B��
                                    Bx��cR  �          A4  ���A�>�33?�\B�L����A�>�{?�  B�L�                                    Bx��q�  Z          A5����A\)>\?�z�B����A\)>�p�?�{B�                                    Bx����  �          A4������A��=�?(�B������A��=���?�B��                                    Bx���D  T          A4����ffA  >�\)?�33B����ffA  >�  ?��B��                                    Bx����  �          A5G����HA��>\?�B�\���HA��>�33?��
B�\                                    Bx����  
�          A3���ffAff>aG�?���B����ffA�\>B�\?uB��                                    Bx���6  �          A3\)����A�
=��
>���B�#�����A�
=#�
>B�\B�#�                                    Bx����  �          A2=q���
A
=>�G�@  B����
A
=>Ǯ@G�B�q                                    Bx��؂  �          A3
=���A	?L��@�33CY����A	?@  @w�CW
                                    Bx���(  �          A3�
��\)AQ�?B�\@x��C � ��\)Az�?5@g�C z�                                    Bx����  T          A4  �ᙚA�
?(�@EC �f�ᙚA�
?��@333C ��                                    Bx��t  �          A4z����
A�>Ǯ?�p�C0����
A�>���?�C0�                                    Bx��  T          A4Q�����A��=�?!G�C ������A��=u>�z�C ��                                    Bx��!�  �          A3�����A��>���?\Ch�����A��>k�?�33Ch�                                    Bx��0f  �          A3���
=A	p�>�{?޸RC���
=A	��>��?���C
=                                    Bx��?  T          A4Q��߮Ap�=��
>ǮC Q��߮Ap��#�
�#�
C Q�                                    Bx��M�  �          A3����A�
����B�\C ����A�
�u���HC �                                    Bx��\X  �          A3�
��ffAG�=�?
=C :���ffAG�<��
=���C 8R                                    Bx��j�  �          A4z���Aff�
=q�0��B�����A=q�&ff�S33B��                                    Bx��y�  �          A4����\)A�ÿL������B�\��\)A�׿h����z�B��                                    Bx���J  "          A2�H��\)A�\�������B��
��\)A�\�\��
=B��)                                    Bx����  �          A2ff��{AG�?#�
@S33B�����{Ap�?�@+�B��{                                    Bx����  (          A1p���  A33>#�
?Tz�B���  A33<�>��B�                                    Bx���<  �          A0����
=AG����ÿ޸RB鞸��
=AG�����RB��                                    Bx����  �          A2ff����A$Q�
=�C33B�=q����A$(��@  �vffB�G�                                    Bx��ш  
�          A1p��j�HA&�\�!G��Q�B����j�HA&ff�L����(�B��
                                    Bx���.  �          A1���
A$�þ8Q�h��Bߔ{���
A$�þ�33���Bߙ�                                    Bx����  �          A1���}p�A%�=��
>���B��
�}p�A%녽��Ϳ�B��
                                    Bx���z            A1���z�A!�?��
@�z�B��f��z�A"=q?���@��RB���                                    Bx��   
.          A1p���Q�@�p�@*=qA^�HCff��Q�@��R@!G�AR�RC:�                                    Bx���  T          A0Q��\(�A���Y�����HBڮ�\(�A�
�e���
=B��                                    Bx��)l  
�          A.ff�g
=A!��������B�Q��g
=A!G���G��׮B�ff                                    Bx��8  "          A1���
=A�;����C &f��
=A�׿��@  C .                                    Bx��F�  �          A1��B�\A-녿����{B��ÿB�\A-p����\��z�B���                                    Bx��U^  �          A733?�
=A3�
�����G�B���?�
=A333�������B��                                    Bx��d  T          A6�R@^{A(z��(���VffB�� @^{A'��8Q��iB�L�                                    Bx��r�  T          A733@�
=A��(���UBx
=@�
=A  �7
=�hQ�Bwp�                                    Bx���P  �          A6{@�\)A�5�h(�Bj(�@�\)A���C�
�zffBiff                                    Bx����  �          A1p�?��HA-G����
�RB���?��HA,�׿�
=� ��B��=                                    Bx����  �          A/�
>�A-�������\B��>�A,z��\)�G�B��f                                    Bx���B  �          A1��?��
A(  �˅��B���?��
A'33������\B��R                                    Bx����  �          A1p����HA.{�B�\��G�B�����HA-녾��\)B��
                                    Bx��ʎ  �          A0zῸQ�A,  >�\)?��RB�G���Q�A,  �#�
��B�G�                                    Bx���4  �          A/�?�G�A+\)�s33����B�L�?�G�A*�H��  �ҏ\B�B�                                    Bx����  �          A.�H�   A(z�>aG�?�
=Bɀ �   A(zὣ�
���B�z�                                    Bx����  T          A-��&ffA&�\?O\)@�=qB�{�&ffA&�H?�\@.{B�
=                                    Bx��&  �          A,  �o\)A=q?�@8Q�B�k��o\)Aff>u?�ffB�\)                                    Bx���  �          A+\)���A��@(��AeB�����A�H@Q�AN{B�                                    Bx��"r  
�          A+33��z�A�H?�z�@ȣ�B��{��z�A\)?aG�@���B�aH                                    Bx��1  �          A*�R��\@���?��@�p�C@ ��\@��\?Y��@��\C#�                                    Bx��?�  
�          A*=q���AG����H���C����A zῼ(��G�C@                                     Bx��Nd  �          A)G����A녿����ffB�� ���A�������B���                                    Bx��]
  �          A*�\���
A(�?�@8Q�B�ff���
AQ�>W
=?��B�W
                                    Bx��k�  N          A)����
A
=��ff�  B������
A
{���"=qB�.                                    Bx��zV  
�          A-G���{Az��AG���(�B����{A�R�S33���HB�B�                                    Bx����  T          A,z���ffA �׾�33��C�{��ffA Q�#�
�^�RC��                                    Bx����  "          A*{�أ�A=��
>�
=C�{�أ�A�aG���C�
                                    Bx���H  �          A*�R� (�@��Ϳ�p���(�CB�� (�@�33��p��   CxR                                    Bx����  "          A,�����R@���33��ffC
8R���R@�녿�z��{C
s3                                    Bx��Ô  �          A,z��\)@��
���$��C޸�\)@�������7\)C:�                                    Bx���:  "          A,  �z�@����R��z�C{�z�@�����&ffC&f                                    Bx����  
�          A-���@�ff>8Q�?s33C�H��@��R���
��Q�C��                                    Bx���  �          A.=q��H@�Q쿯\)��\CJ=��H@�ff�У���
C��                                    Bx���,  �          A.�\�z�@�p���Q��C��z�@��Ϳ!G��W
=C�                                    Bx���  
(          A.�H�{@ۅ?\)@<(�C��{@�(�>�=q?�
=C�)                                    Bx��x  	�          A0z���@���?ٙ�A�\C���@��H?�33@��HCB�                                    Bx��*  �          A1G���@��\?�=q@�p�C����@��
?=p�@y��C�\                                    Bx��8�  T          A/\)���H@��
@*=qAa��C	Ǯ���H@�\)@ffAF�\C	Y�                                    Bx��Gj  �          A0  �
=q@ҏ\?�  A=qC���
=q@���?��H@�33Ck�                                    Bx��V  
�          A0z��
ff@�?��@�  CQ��
ff@�\)?��@�p�C
                                    Bx��d�  �          A/\)��R@�Q�?��@�  C=q��R@��?Tz�@��C\                                    Bx��s\  Z          A.�R��@�{?ǮA  C����@�  ?��\@�
=C�                                    Bx���  
�          A/���@�Q�?s33@��CG���@љ�?#�
@XQ�C�                                    Bx����  �          A0�����@�ff@�\A,��CW
���@�G�?ٙ�A(�C
��                                    Bx���N  T          A0����p�@�=q�(��L(�CO\��p�@��ÿxQ���33Cu�                                    Bx����  �          A1����@��=�\)>ǮC
�
���@�Q쾏\)���HC
�)                                    Bx����  �          A2=q�=q@�
=>��@=qCG��=q@�=�G�?��C8R                                    Bx���@  
�          A3��ff@���aG�����CǮ�ff@�z����E�Cٚ                                    Bx����  �          A3�
��
=A�
�\����C5���
=Aff��Q��\)C�                                    Bx���  
�          A3
=���\A����Q���33B��
���\A��{�ң�B�p�                                    Bx���2  "          A4����G�A����
��{B�����G�A
=q���\��z�B�Q�                                    Bx���  �          A3\)���RA=q�aG����B�aH���RA
=�\)��  B��{                                    Bx��~  �          A2=q���A
{��p���  B�����Aff��(��ď\B�                                      Bx��#$  �          A2ff����A(���\)�ׅB�33����@�\)����z�B�
=                                    Bx��1�  �          A2�\��p�A���G���
=B����p�A���Q���B�u�                                    Bx��@p  "          A4z���A��P  ���C @ ��Az��n{��  C �f                                    Bx��O  "          A5���z�A�\�h����(�B�=q��z�A����Q���33B���                                    Bx��]�  �          A4  �׮A�\�Q���{B�8R�׮Ap���=q���B���                                    Bx��lb  �          A3�����Aff���
����C\)����A=q��\�&ffCh�                                    Bx��{  �          A1���AQ�p����33C����A33�����
C�                                    Bx����  "          A2ff����A	���n{��G�C@ ����AzῸQ���Cz�                                    Bx���T  "          A2{��p�AG���z���  B�#���p�A�
��Q��  B��                                    Bx����  �          A333��(�A33�������B�  ��(�A녿�
=�	B��=                                    Bx����  �          A3
=�ٙ�AG��fff��33B�(��ٙ�A(���Q�����B���                                    Bx���F  �          A3���ffAQ쿋����B��\��ffA
=��z��Q�B�\                                    Bx����  
�          A3\)�љ�AG��&ff�Tz�B����љ�AQ쿜(��ǮB���                                    Bx���  T          A2ff��Q�Ap��.{�^�RB�����Q�Az῞�R��(�B�8R                                    Bx���8  	�          A2�\����A
=������CE����A����R�$��C��                                    Bx����  
�          A3
=��\A�R�E��~{C�H��\A����ff��ffC�                                    Bx���  T          A1���=qA�   �$z�C5���=qA�ÿ�ff���CaH                                    Bx��*  �          A2�R�陚A33���
��z�C�\�陚A�\�c�
����C�3                                    Bx��*�  �          A2�H��{A���L�Ϳ��
C�R��{A�ÿB�\�y��C�
                                    Bx��9v  N          A2ff���@�����
��
=C  ���@��H��R�J�HC
                                    Bx��H  
�          A2ff����A��������C�����A(����	CaH                                    Bx��V�  �          A3
=��(�A녽�G��z�Cff��(�Ap��0���c�
C�                                     Bx��eh  T          A3
=��=qA33���.{C�f��=qA�H�!G��Mp�C��                                    Bx��t  "          A3\)���A�
���;��HC�
���A\)�333�dz�C��                                    Bx����  "          A2�H��G�A녾��&ffB��H��G�Ap��B�\�z=qB�{                                    Bx���Z  �          A2�H���A�<#�
=uB������A
=�&ff�W
=B���                                    Bx���   T          A2�R��{A��>\?�
=B��)��{A�;�z´p�B��)                                    Bx����  �          A333����A=q�B�\�xQ�B�����A���\(����RB�=q                                    Bx���L  T          A2ff��
=AG�>�  ?��C�3��
=A���p���
=C��                                    Bx����  T          A1���{@��?��A  C�q��{A��?��\@ҏ\C��                                    Bx��ژ  "          A2ff�У�A\)>#�
?Tz�B�
=�У�A33���,(�B��                                    Bx���>            A2�\��(�A�׿�p���HB�����(�A{�(��J{B��=                                    Bx����  
�          A3\)����A
=�Mp����B������A�R�{����HB�                                    Bx���  �          A1����ffA\)�����B�W
��ffAG���-�B�                                      Bx��0  O          A2�\��A
=�h����p�B�����Ap���Q��\)B�G�                                    Bx��#�  Y          A2�R���A!��Q����RB�p����A���\)�G�B��)                                    Bx��2|  T          A3
=���A#\)���
���HB�B����A!��ff�,��B���                                    Bx��A"  T          A333���
A%G������أ�B߅���
A#
=�	���1�B�\                                    Bx��O�  O          A4Q����A%녿�  �˅B߮���A#�
��+33B�8R                                    Bx��^n  
�          A3\)��G�Ap���z����B�Q���G�A�H����F�RB�Q�                                    Bx��m  O          A4  ��\)A��-p��`Q�C�)��\)@�=q�W���Q�C�                                    Bx��{�  
�          A3\)���HA�
�ff�,��B�#����HAz��:=q�p��B�(�                                    Bx���`  "          A3����HA{�p�����
B��
���HAQ��p���B�u�                                    Bx���  �          A4(���Q�A&=q�\)�0��B��{��Q�A%p����\���RB�Ǯ                                    Bx����  
�          A3�
���A{?
=q@0��B����A=q���R��=qB�z�                                    Bx���R  "          A3����A�?�@9��B��)���A��������
B���                                    Bx����  "          A4(���ffA�\?333@c�
B�k���ffA�H�.{�aG�B�Q�                                    Bx��Ӟ  �          A4(���ffAG�?�33A  B���ffA\)?���@�\)B�L�                                    Bx���D  "          A4Q���{Ap�@-p�A_33B��
��{A��?���A\)B��
                                    Bx����  �          A2�\��  AQ�?�p�A#�B�8R��  A�\?���@�z�B�\                                    Bx����  "          A2=q���RA(�?�  A��B����RA{?aG�@��B�\                                    Bx��6  �          A1��33A33@�A0��C\��33A	��?��@�
=C �=                                    Bx���  �          A2{����A��?�G�AC������A�\?}p�@��
C^�                                    Bx��+�  �          A2�\��z�@��
?�A
=C���z�@���?��H@ǮCxR                                    Bx��:(  �          A2�H�(�@��H@p�A6�RC\�(�@أ�?˅A�HCQ�                                    Bx��H�  T          A1�z�@أ�@	��A2ffC�
�z�@�{?�  @���C��                                    Bx��Wt  �          A1p����A�?�A Q�C �\���A	?�{@���C W
                                    Bx��f  �          A3���z�A$��?�33@�B�����z�A&=q>��@ffB�u�                                    Bx��t�  �          A4  ��G�A&ff?�  @�(�B�p���G�A'�>u?��RB�.                                    Bx���f  �          A3�
��z�A!p�?�{AQ�B�aH��z�A#33?!G�@N{B��                                    Bx���  �          A3
=���HA!�?���@�ffB�  ���HA"�R>�@
=B䙚                                    Bx����  �          A2=q��{A�R?��HAG�B�L���{A��?G�@�  B��                                    Bx���X  �          A3
=��Q�A��@G�A&�\B�W
��Q�A33?�\)@�
=B�k�                                    Bx����  �          A2=q��(�A@{A8��B��3��(�Az�?�ff@�\)B��R                                    Bx��̤  �          A0�����Ap�?�33A	C@ ���A\)?O\)@�ffC�
                                    Bx���J  �          A1����\@�33?�A�C�)���\@�\)?fff@�p�C^�                                    Bx����  "          A0������A(�?�ffA�C�f����A�?333@h��C��                                    Bx����  �          A1��ffA	�@�A(z�B�����ffA��?���@��B���                                    Bx��<  �          A2=q��
=A{@��A;\)C��
=A��?���@�(�C!H                                    Bx���  �          A1p�����A�@'
=AZ�\C0�����A��?�p�A(�C p�                                    Bx��$�  �          A1p���
=A@1G�Ag�C ���
=A	?��AQ�B��                                    Bx��3.  �          A1G���  A\)@7
=Ao�
B�\��  A\)?���A"�\B�k�                                    Bx��A�  
�          A0(���{A	G�@^{A�G�B�ff��{Aff@"�\AU�B�p�                                    Bx��Pz  �          A/
=��ffA\)?�=qA
=B����ffA��?aG�@�B�
=                                    Bx��_   �          A/
=���RAff��\)���HB�����RA���G���(�B��{                                    Bx��m�  "          A.�\����A�H��R�Tz�B��H����A�ÿ�=q��
B��{                                    Bx��|l  �          A/
=���RA=q>�ff@
=B�8R���RA{����8Q�B�B�                                    Bx���  �          A/��ӅA ��@��A9p�CaH�ӅA�
?��R@�ffC �q                                    Bx����  T          A.{���@�@XQ�A�{C�����@�Q�@$z�A\(�C:�                                    Bx���^  �          A-���Q�@�  ?���A&=qC�3��Q�@�p�?���@�  CJ=                                    Bx���  "          A/\)���HA��>�@=qC �����HAz�   �&ffC �R                                    Bx��Ū  �          A.�R�޸RA��?}p�@�\)C���޸RA�=L��>��C                                    Bx���P  �          A.ff��\)A (�?���@�(�C���\)A>�(�@�\C��                                    Bx����  �          A-����HA	�=�G�?z�B������HA	��^�R���
B��                                    Bx���  �          A-��ȣ�A����(��B��
�ȣ�A  �������B�G�                                    Bx�� B  �          A.{���A\)�aG���Q�B��
���A
{��p��љ�B�aH                                    Bx���  �          A-���A ��@�HAN�\CB����AQ�?��H@���CxR                                    Bx���  T          A-��(�A ��@�
AEG�C�=��(�A(�?�{@�ffC�=                                    Bx��,4  T          A-G����A
=@=qAMp�C����A�R?�
=@�=qC T{                                    Bx��:�  �          A,�����A ��@�RA?\)CE���A  ?�G�@׮C��                                    Bx��I�  
�          A,����  @�R@
=qA8Q�C���  @�p�?�  @��CY�                                    Bx��X&  
�          A,����{@���@B�\A�G�C&f��{@�R@
=qA9G�C�                                    Bx��f�  �          A,����p�@�
=@p�A=��C����p�@�?�G�@׮C&f                                    Bx��ur  "          A,����R@��
?У�A33C^���R@�Q�?333@l��C�
                                    Bx���  �          A,z����H@�G�?��
@ٙ�C0����H@�z�>�33?���C��                                    Bx����  T          A,z���\)@��?��@���C
)��\)@�>L��?��C	��                                    Bx���d  �          A-G���\@�=q?!G�@S�
C���\@��H��  ���C�                                    Bx���
  T          A-����G�@�Q�?8Q�@tz�C33��G�@�G���  ��ffC
                                    Bx����  T          A,���{@�33    <��
C�f�{@ٙ��Y����33C�                                    Bx���V  �          A,Q��33@�\)�J=q��
=C��33@�G��p����(�C�                                    Bx����  "          A-p����@�=q���H��G�Cn���@��H�{�=��CxR                                    Bx���  �          A-G����@�?޸RA��C+����@��?J=q@�{C��                                    Bx���H  �          A.ff��p�A�H@��HA�ffB���p�A
=q@@��A�Q�B��
                                    Bx���  �          A-G���\)A�@Z�HA��B�ff��\)A
=@z�AF�RB�\                                    Bx���  T          A-�����HA�@j�HA�ffB��
���HAQ�@#�
AZ�RB�Q�                                    Bx��%:  �          A-��  @�@aG�A��CY���  A�@p�AR{C �                                    Bx��3�  �          A-��ۅ@�  @L��A�\)C���ۅA��@
=qA7�
CE                                    Bx��B�  �          A.ff���HA�@Q�A��RB�G����HAG�@
=A333B�8R                                    Bx��Q,  T          A.�\���HA(�@%�AZ�\B����HAQ�?�\)@�{B�(�                                    Bx��_�  
�          A.=q���
A��@ ��AUB�u����
A��?���@�\)B���                                    Bx��nx  �          A.�H���
A	�@=p�A}p�B�z����
A
=?��
A�B��\                                    Bx��}  �          A/���{Aff@@  A~=qB�(���{A�?޸RA�\B�z�                                    Bx����  
�          A/�
���A�@;�Aw\)B�\���A��?�Q�A��B�L�                                    Bx���j  �          A/33��p�A(�@HQ�A���B�����p�A��?�A��B�Q�                                    Bx���  �          A/33���HA
=@333An{B������HA�
?ǮA\)B�L�                                    Bx����  T          A/����A��?ǮA�B�  ���A
=>�=q?�
=B�Q�                                    Bx���\  �          A/
=���
A
�R@�\AC33B�aH���
A�\?�=q@�ffB���                                    Bx���  T          A/
=��{A�
?��
AffB��f��{Aff?�@/\)B���                                    Bx���  �          A/\)��G�A��?��A�B����G�A�H>���?˅B�Q�                                    Bx���N  �          A/�
�ָRA�?�=qA��C \)�ָRA	�>\?�p�B���                                    Bx�� �  �          A.�H����@�G����
��
=C}q����@�ff����p�C�\                                    Bx���  T          A/33��  @��R��\)��p�C&f��  @�33��\)��{C�{                                    Bx��@  �          A0  ��33A�@\)A<��B�33��33A\)?p��@�B��f                                    Bx��,�  �          A/�
���A
=q@
=A2�RB�=q���A?\(�@���B��                                    Bx��;�  �          A0Q���
=A��@1�Ak�
B�  ��
=A�?��
A z�B�\                                    Bx��J2  
�          A0Q���=qA?�
=@�\)B�����=qA�>\)?8Q�B�B�                                    Bx��X�  �          A/�
��G�A��@�AIp�B�����G�A��?�{@�33B�{                                    Bx��g~  
Z          A0Q����
A�@'�A\��C T{���
A
�\?���@�RB�                                    Bx��v$  
�          A0Q��ٙ�A��@�A?
=C0��ٙ�A	p�?��
@�(�C aH                                    Bx����  �          A/���\)A
=?�  A�C �{��\)A	��>��H@%C 
=                                    Bx���p  �          A/���=qAp�?��R@�G�B����=qA\)>.{?^�RB�\)                                    Bx���  �          A/\)��ffA��?��RA'�B�\)��ffA�
?(��@^{B�(�                                    Bx����  �          A/
=��ffA
=@ffAF�RB���ffA
=?��
@�p�B�z�                                    Bx���b  �          A0Q�����A��@$z�AX��B������A�?�p�@�p�B�ff                                    Bx���  �          A1G���33A�?�G�@���B�.��33A(���
=��B���                                    Bx��ܮ  �          A1����HA��@�AK33B��3���HA��?��
@��
B�8R                                    Bx���T  �          A0������A�@   AQ�B�������Ap�?��@�B�p�                                    Bx����  �          A1����A��@k�A�{B����AQ�@{A9�B�=                                    Bx���  �          A0������A�R@*=qA_�B�������A�?��\@�33B�=q                                    Bx��F  "          A0(���\)A
{?���A$(�B�����\)AG�?��@J=qB��\                                    Bx��%�  "          A/��{@�\��=q���C�=�{@޸R��{��CJ=                                    Bx��4�  �          A.�R���R@�{�\)�=p�C����R@�Q���H�Q�C	��                                    Bx��C8  �          A/\)�{@�
=������C
k��{@޸R�
�H�7\)Cn                                    Bx��Q�  "          A0(���  A �;��+�C����  @������Cff                                    Bx��`�  �          A/
=��33@��\>�G�@z�C#���33@����J=q��p�C=q                                    Bx��o*  �          A/�
���@�{?8Q�@q�Ck����@��R�
=q�5�C^�                                    Bx��}�  �          A-����p�@��R@p��A�C����p�A33@\)AS�C!H                                    Bx���v  �          A-�����@�ff@=qAMG�C\)����@��?�z�@�33CQ�                                    Bx���  �          A-G����@�{?\A��C�3���@��\>�=q?�Q�Ck�                                    Bx����  �          A,��� ��@�33?��@���C
��� ��@���Q���HC
B�                                    Bx���h  �          A,z���33@��?!G�@W
=C:���33@�녿#�
�Y��C=q                                    Bx���  �          A,����Q�@���?��
@�
=C����Q�@��R�����z�C^�                                    Bx��մ  �          A,Q���@�?�ff@��C����@�  �.{�k�C��                                    Bx���Z  �          A,Q���\)@�?�G�@�\)Cff��\)@�R=#�
>W
=C�q                                    Bx���   �          A-����@�z�?��@ۅC�=���@�  <��
=�G�C#�                                    Bx���  �          A-������@��H?�{A��C������@�  >\@   C�q                                    Bx��L  O          A-�����R@���@8��Ax  C	�H���R@��?�Q�A�
C�                                    Bx���  Y          A,����p�@�33@&ffA_�Cff��p�A�H?�G�@�\)C@                                     Bx��-�  
�          A+���ff@�p�?��A"=qC
�R��ff@�z�?5@q�C

                                    Bx��<>  T          A*ff��ff@߮@(�A>=qC	�=��ff@�Q�?}p�@�33C�R                                    Bx��J�  �          A*{�33@Ϯ?��
A�HC�)�33@�ff?+�@g
=C��                                    Bx��Y�  �          A*ff�z�@��?k�@��C�)�z�@��
�L�Ϳ��CY�                                    Bx��h0  T          A*�\�z�@�p�?�
=A�C5��z�@Ӆ?�@ECc�                                    Bx��v�  T          A*�H�	p�@�33?У�Az�C���	p�@�G�?�@C33C�=                                    Bx���|  �          A*�\�z�@���@   A-p�C���z�@���?���@���Ck�                                    Bx���"  �          A+
=�=q@���@!�A[
=CǮ�=q@�Q�?�33A=qC\                                    Bx����  "          A*�H���@��@{AV�\C
=���@��H?��RA ��Cu�                                    Bx���n  �          A+
=�ff@��@�A/33C�q�ff@��\?��@��HC�                                    Bx���  T          A+
=�G�@��
@*�HAg�C���G�@�Q�?ٙ�AffC�                                    Bx��κ  T          A*�H�	�@��R@�\A0  C\)�	�@�
=?xQ�@��RC0�                                    Bx���`  �          A*=q��@��H@  ADQ�Cff��@�z�?��R@׮C�q                                    Bx���  �          A)�z�@�=q@AG�A�z�C���z�@���@�A5p�C�{                                    Bx����  �          A(���z�@mp�@|(�A��C���z�@�33@N�RA��HCJ=                                    Bx��	R  �          A'�
�
�R?G�@�G�B�C.�
�
�R?��@��B �\C(O\                                    Bx���  �          A)��\)?��@���A�z�C(k��\)@*=q@�z�A��HC#޸                                    Bx��&�  �          A)���@?\)@��\A�=qC!ٚ��@mp�@n�RA��Cٚ                                    Bx��5D  T          A(����@`  @mp�A��HC8R��@��@A�A�p�C                                      Bx��C�  �          A)��
=@�33@vffA���CW
�
=@�\)@C33A�(�C!H                                    Bx��R�  �          A)���R@L��@g
=A�p�C!8R��R@s33@>{A�z�C
=                                    Bx��a6  �          A)G��
=@h��@9��A�  C��
=@�33@��A@��C�                                    Bx��o�  �          A)G��z�@7
=@~�RA�Q�C"�H�z�@a�@Y��A��C(�                                    Bx��~�  �          A)p���@�\)@9��A33C^���@�@�A6=qC�                                    Bx���(  �          A*{�ff@��@Q�A9�C���ff@�G�?���@�p�C�                                    Bx����  �          A*�R�Q�@���>��?Tz�C���Q�@�ff��  ���C��                                    Bx���t  �          A+33���@�{?^�R@�C޸���@�Q���Ϳ�\C��                                    Bx���  �          A*�\�33@�G����
���
C^��33@�p����\�ڏ\C�)                                    Bx����  �          A*�R�	p�@�G�>.{?c�
C�	p�@�
=��  ����C�                                    Bx���f  T          A+
=�@�
=��z��ffCB��@�녿�p�� ��C��                                    Bx���  T          A,  �(�@θR@Q�A7�C���(�@׮?c�
@�Q�CǮ                                    Bx���  �          A+���\@�p�@L��A�
=C���\@���?��RA+�C�\                                    Bx��X  �          A+��\)@ə�?���A�CW
�\)@���?#�
@[�CT{                                    Bx���  �          A,z���@��?�Q�@��C��@�=q>�  ?��C\                                    Bx���  �          A,z���@�\)    <��
C����@Å���ȣ�C                                      Bx��.J  �          A,  ��\@��?���A�C
=��\@��>��@�C8R                                    Bx��<�  "          A,����\)@�  >B�\?�  C�R��\)@�zῥ����
C\)                                    Bx��K�  �          A+\)���@�33�+��h��C	�R���@��H� ���/�C
�                                    Bx��Z<  �          A,  ��@�
=�fff���RC{��@���z��L��CL�                                    Bx��h�  
�          A+\)��{A�
�B�\���B��
��{@�{���Pz�B��H                                    Bx��w�  �          A*�\����A��#�
�aG�C u�����A �׿޸R���C
                                    Bx���.  �          A*{��=qA�׿�G��ۅB�Ǯ��=qA=q�8Q��}�B�k�                                    Bx����  
�          A*{���H@��?E�@�{C�\���H@���J=q���C��                                    Bx���z  �          A*{����@�\)?��@�\C������@�33���(��C#�                                    Bx���   �          A(����@Ǯ?h��@���C�
��@�G����
��\C�
                                    Bx����  "          A*=q���
@��
@�ffA��
B�����
A��@%�Ab�RB��q                                    Bx���l  "          A*�\�θR@�@%�AfffC
=�θRA ��?��\@���C                                     Bx���  T          A+\)��G�A ��?�ff@�{C���G�A�\)�@  C�                                    Bx���  �          A+���(�A  @   AW�B�ff��(�A	G�?Q�@���B�=q                                    Bx���^  �          A*�\���R@�@U�A�Q�B������RA
=?�Q�A��B�u�                                    Bx��
            A+��ƸRA  �c�
���B���ƸR@��� ���`  C !H                                    Bx���  Y          A+\)���AG�?5@u�C �����A�Ϳu����C �                                     Bx��'P  T          A+33����@�>aG�?�z�C	������@�\��(���=qC
#�                                    Bx��5�  �          A+33��G�A
=�\)�E�B��=��G�A�
��=q��C p�                                    Bx��D�  �          A*�\����A\)���:�HB�����A(���=q�"{B�L�                                    Bx��SB  �          A*{����A33�L�;�\)B�L�����A(������"ffB�p�                                    Bx��a�  �          A*{��(�A ��=L��>�\)C���(�@�z�������C�                                    Bx��p�  "          A)��G�A
�H?\)@AG�B�����G�A	���H���HB�
=                                    Bx��4  �          A'����@�\@�AN{C���@���?J=q@�{C��                                    Bx����  �          A&ff��ff@�  @7�A��RC����ff@θR?˅A�C�                                    Bx����  �          A%�����
@ʏ\@1�Az�RCG����
@�Q�?�
=@�
=C
ff                                    Bx���&  
�          A%p�����@��
?��A!p�C������@Ӆ>�G�@�RC�)                                    Bx����  
�          A&{�33@���O\)����C��33@��R��\)�)�CL�                                    Bx���r  "          A"ff�޸R@�(�@�AY�CQ��޸R@�
=?fff@�(�C�                                    Bx���  
�          A"ff��  @�\@%�AlQ�C����  @��R?�G�@�\)C�                                    Bx���  
�          A"ff�Ϯ@�p�@\)AM�C+��Ϯ@�\)?(��@n{C
=                                    Bx���d  
�          A"�H��z�A��J=q��
=B�  ��z�A��%�l(�B��                                    Bx��
  �          A#33��G�A����\)�{B��
��G�A��[���33B��)                                    Bx���  �          A$(���
=AG���\�8(�B�����
=A33�x����33B�                                    Bx�� V  T          A$Q����A�R�����B垸���A	��a���33B�\)                                    Bx��.�  �          A#
=����A��>��@z�B�\����A
=��33���B�33                                    Bx��=�  Y          A#
=�Ϯ@��R=�\)>ǮC)�Ϯ@񙚿Ǯ�=qC�3                                    Bx��LH  T          A#\)��A33?   @8Q�B���A	������\)B�=                                    Bx��Z�  "          A$(���  A��>�@*�HB����  A�H��  �\)B垸                                    Bx��i�  T          A$�����A��n{��p�B������A�\�8����ffB��                                    Bx��x:  �          A$������AG�?.{@s33B�k�����AQ쿢�\���HB�q                                    Bx����  T          A#���\)Az�?�
=A.�HB�{��\)A�
=�\)>�p�B���                                    Bx����  T          A#
=��AG�����W
=B��{��A����
=�0Q�B�
=                                    Bx���,  �          A"�H��  @ٙ�@�A8Q�C�)��  @�=q?�@;�C�R                                    Bx����  �          A$�����R@�\)@�A:�HC�3���R@أ�?#�
@c�
C
��                                    Bx���x  �          A$z���\@��@�AZ�\C	���\@�p�?fff@�G�C��                                    Bx���  �          A%G�����@�(�@ffA;�
C{����@��>�ff@!G�C                                    Bx����  �          A%����A{?�@:�HB�
=���Azῼ(��=qB�{                                    Bx���j  �          A%�����A��?��A(Q�B�Q����AQ켣�
����B��                                    Bx���  �          A%�����A	G�@\)AI�B�33����A>��
?��
B�=                                    Bx��
�  T          A$Q���  A��?�A%�B�� ��  A�
���
����B�8R                                    Bx��\  �          A$z���G�A�R@A<Q�B�����G�A�R>�  ?���B���                                    Bx��(  �          A$Q���p�@��\?˅A
=C\)��p�@���.{�k�C ��                                    Bx��6�  "          A$������@��\>�G�@p�CY�����@�������\)C��                                    Bx��EN  
�          A&ff���A�?���@�(�B�33���A{�+��l��B���                                    Bx��S�  
�          A&�\��G�A ��?0��@r�\C���G�A   ��{��z�CE                                    Bx��b�  "          A'���G�Az�?�@�B�p���G�AG��0���s33B��                                    Bx��q@  �          A'����A��?!G�@]p�B�ff���A  ��p��أ�B��
                                    Bx���  �          A(  ��A
=q>�
=@�\B�����A(��\���B��q                                    Bx����  Y          A(�����A\)�L�;�\)B�p����A���Q��)p�B��                                    Bx���2  �          A(Q���  A�H?\)@A�B�#���  Ap������33B��R                                    Bx����  �          A(Q��ϮA  ?0��@qG�C .�ϮA
=��
=��p�C ^�                                    Bx���~  �          A)G���(�A=q?L��@��HB��3��(�Ap���(�����B���                                    Bx���$  �          A)�����A�?�@Dz�B�p�����A녿�(��   B�
=                                    Bx����  
(          A)��|(�A��8Q�s33B߽q�|(�AQ�����P��B���                                    Bx���p  T          A)G�����@�z�?z�@Y��C������@�\��ff��C.                                    Bx���  �          A)����H@^{@{AB�\C G���H@xQ�?�\)@�C&f                                    Bx���  �          A)p��ff@�Q�?�=qA�CL��ff@�G�?&ff@`  C�                                    Bx��b  "          A(z��{@��?�
=A)�C!H�{@��?+�@i��C��                                   Bx��!  �          A(�����@�(�?��RACc����@�=q>#�
?c�
C�                                   Bx��/�  "          A(Q��p�@��R?�\)A�CQ��p�@�>���?��CE                                    Bx��>T  �          A((���\@���?�AffC\)��\@���?��@?\)C�                                    Bx��L�  �          A'\)�z�@���?�(�A�C�H�z�@���>�{?��C�=                                    Bx��[�  �          A(����@�=q@�RADQ�C� ��@�?Tz�@�G�C�H                                   Bx��jF  
�          A(  ���@��@`  A��C����@θR@�
A5G�C�                                   Bx��x�  T          A(Q����@�33@;�A��C	=q���@��H?��\@�CB�                                    Bx����  �          A'\)� (�@��H@P  A�33C�� (�@�?�A�C=q                                    Bx���8  "          A&�R��33@���@K�A��RCn��33@Ӆ?ٙ�A=qC��                                    Bx����  T          A&�R��@���@:=qA�=qC���@��?��@���C	�=                                    Bx����  "          A&{�陚@߮@ffAQ�CG��陚@��H?(��@i��Cٚ                                    Bx���*  
(          A&�\�޸R@��@{AEp�C+��޸R@�\)>�(�@
=C�q                                    Bx����  �          A&�\�أ�@�G�@Q�AS�C��أ�@�z�?��@AG�C��                                    Bx���v  �          A'
=�Ӆ@�@��AU��C���ӅA z�?�@<(�CxR                                    Bx���  
�          A&ff��{A��@�\A5p�B�����{A��=��
>�ffB��                                    Bx����  �          A&�H��G�@陚?�  A�C����G�@�Q�#�
���
C(�                                    Bx��h  T          A'
=�ǮA=�\)>���B�k��ǮA=q��33�(Q�B��H                                    Bx��  �          A"ff��z�A�ÿ�ff��
=B�Q���z�@�=q�?\)����B��\                                    Bx��(�  "          A"ff�|(�A���
=�@��B�  �|(�A����ff��z�B�                                    Bx��7Z  "          A#33��p�A\)��\�8z�B���p�A������{B�=                                    Bx��F   "          A%���@���@AC
=C	k���@��H>��@/\)C�                                    Bx��T�  �          A%G���ff@�ff@S�
A�\)CaH��ff@�=q?���A%G�C�                                    Bx��cL  �          A$(��陚@��
@2�\A}�C	���陚@�33?���@��HC��                                    Bx��q�  T          A$  ��  @�Q�@E�A�=qCG���  @�=q?�ffAC
                                    Bx����  �          A$����@��
@<(�A�
=C�
��@�z�?���@���C	^�                                    Bx���>  "          A$  �(�@��H@8��A��Cc��(�@�z�?��A(�C��                                    Bx����  �          A%���p�@ڏ\?��@C�
C
L���p�@�Q쿎{��ffC
�
                                    Bx����  T          A#����@��H@  AL(�C
W
���@�{?
=@U�C�H                                    Bx���0  T          A!p���\)@�\)?���A2�HC	���\)@�  >�=q?��
C�                                    Bx����  "          A   ��@��?+�@tz�C
�\��@У׿p�����C
��                                    Bx���|  T          A���=q@�  ���:�HCG���=q@�Q�����+�CE                                    Bx���"  "          A{��Q�@�33?��@�p�Cff��Q�@�ff���5C                                      Bx����  
�          A�\��33@�R@��AS�C^���33@�G�>�G�@"�\C
                                    Bx��n  �          AG�����@�ff<��
=�G�C#�����@�\)��ff�*{C�q                                    Bx��  T          A�H��G�@��@I��A�33C
=��G�@�\)?�A4Q�C�f                                    Bx��!�  
�          A���@���@!�A{�C���@��?���@�
=C��                                    Bx��0`  �          AQ���z�?�  @��B*��C&�\��z�@<(�@�Q�B\)CO\                                    Bx��?  �          A(��`��A z��
=�0��B�.�`��@��Q��{�
B�{                                    Bx��M�  �          A�H=�\)A
=�k�����B�\=�\)@�Q���Q��ffB��f                                    Bx��\R  �          Aff>ǮA  ������Q�B���>Ǯ@����G��){B��\                                    Bx��j�  �          A?�p�A�H�a���Q�B���?�p�@陚��  ���B�\)                                    Bx��y�  "          AG�����A��33�N{B�Ǯ����A�H��ff��\)B�#�                                    Bx���D  
�          A��@j�H@��W����\B��@j�H@Ӆ����	G�Bs�H                                    Bx����  T          A=q���@�\)�����M�B�(����@�p��<����ffBυ                                    Bx����  
�          A\)��ff@Mp�@Mp�A�Q�CY���ff@z=q@�
Ao�
C
                                    Bx���6  
�          A(���=q@��H@33AN�\C	5���=q@���>�@>{C�                                     Bx����  T          A���~{A\)��z��!G�B�L��~{@����n{��Q�B�{                                    Bx��т  �          A��AG�A�H����fffB�p��AG�A���(��i��Bٳ3                                    Bx���(  �          A�\�J=qA\)@
=qAM�B��)�J=qA��\)�Y��B��)                                    Bx����  �          A�\�<��A�@\��A�=qB�\)�<��A��?�z�@�  B�(�                                    Bx���t  �          A���J=qA��@
=AK
=Bۅ�J=qA�;.{��  Bڅ                                    Bx��  �          A�H�fffA��@(��Aw33B�G��fffA�>���?�
=Bޞ�                                    Bx���  �          A#
=�eA�@z�AR�HBފ=�eA(����
��(�B�\)                                    Bx��)f  "          A%p��b�\A��@`��A���B�Ǯ�b�\A=q?�\)@�\)B�\)                                    Bx��8  �          A ���Z=qA  @AG�A�33B�u��Z=qA�?&ff@l(�B۔{                                    Bx��F�  
�          A33�=p�Az�?�
=A<Q�B�8R�=p�A�
��33�
=B�k�                                    Bx��UX  �          A�R��z�Ap�?���A��B�8R��z�A
=�Q���(�B�                                      Bx��c�  T          A"�H��A�@(�A_\)B�\��A!녽��0��B��H                                    Bx��r�  �          A'
=���A ��@��AJ�RB��ᾅ�A$�;�Q�� ��B��                                    Bx���J  T          A/�@ ��A�H@�z�A��B��@ ��A*�\?�p�@���B���                                    Bx����  �          A1����p�A�H�ff�S�B�(���p�A(������޸RBÔ{                                    Bx����  �          A0  �g
=@�G����H�633B�
=�g
=@�=q����g�HC W
                                    Bx���<  �          A$���
=@��H�љ��
=B�{�
=@�p��Q��WQ�B��)                                    Bx����  
�          A)�����@�  ���
�"�B�G�����@���	��[�Bٔ{                                    Bx��ʈ  X          A(�����@�=q��{�;(�Bڞ����@��
��H�qB��                                    Bx���.  "          A&=q���H@�����H�%��B�G����H@�\)�	���`�
B���                                    Bx����  
�          A z�L��A���
�z�B��R�L��@��H��  �J�B��                                    Bx���z  T          A$  @EA
ff�����хB���@E@�ff���H� 33B��                                    Bx��   �          A"�\?�Q�A��������B��3?�Q�@�������G  B�8R                                    Bx���  
�          A���  @��
��\)��B�p���  @������O33Bƨ�                                    Bx��"l  �          Aff��@�(���  �(�B��f��@�=q���\�P�B�u�                                    Bx��1  
�          A�\?O\)@�\��(��z�B��H?O\)@����z��Q�HB�8R                                    Bx��?�  �          Ap�?���@�����33�&�B�aH?���@�����p��_��B�{                                    Bx��N^  �          A=q@\)@�  ���H�7��B�@\)@�{�z��o
=B}�                                    Bx��]  �          AG�@W�@�z�����G=qBg��@W�@P���
�\�v\)B0�                                    Bx��k�  T          A��@7
=@�G����
�C33B~�@7
=@k��	��vz�BP��                                    Bx��zP  �          A��@(�@�Q���=q�I�B�@(�@fff����~�
B_33                                    Bx����  
�          A  @z�@�p�����\�\B�u�@z�@:�H�����BZ                                    Bx����  �          A  @<(�@c�
�  �r�BI@<(�?���=q=qA�                                    Bx���B  T          A�H?\)@����H�+(�B��?\)@c�
��
=�g{B�=q                                    Bx����  
�          A\)����A	�>�p�@�RB�W
����A녿���733B�ff                                    Bx��Î  �          A(����HAG����333B����HAff�1G���  B��f                                    Bx���4  �          A�
�j=qA\)>L��?��HB�q�j=qA����K
=B���                                    Bx����  �          A\)�c33A{<#�
=uB��H�c33Ap��(��^ffB�W
                                    Bx���  
�          AQ��<(�@�\�s�
��  B߮�<(�@�=q���H��B瞸                                    Bx���&  �          AQ��vff@�{�����{B��)�vff@�(���
=�#\)B�8R                                    Bx���            Ap����@����G���p�B̏\���@�  ��  �/��B��                                    Bx��r  
�          A�Ϳ�Q�@�Q��l���ՙ�Bƙ���Q�@�����\)�&�HB�L�                                    Bx��*  �          A  �k�@��\�qG��̣�B�uÿk�@����ff�"�HB�
=                                    Bx��8�  T          A�R��z�@�G���=q�ٮB�uÿ�z�@�{��\)�(�
Ḅ�                                    Bx��Gd  T          A�
��p�@�Q���p�����B�G���p�@�z���Q��.=qB�{                                    Bx��V
  �          A���4z�@�G������G�B�{�4z�@�����  �2=qB�#�                                    Bx��d�  T          A�H�U@���������
B�\�U@�����
=�1Q�B���                                    Bx��sV  	�          A	���hQ�@�����Q���
=B�33�hQ�@����(��${B�=q                                    Bx����  �          A��,��@�G���  � ��B���,��@�����{�9G�B�                                      Bx����  "          Az���\@�  ��33�33B����\@�G�����B33B��                                    Bx���H  T          A33����@�z���=q���B�\����@�\)���
�2�Bр                                     Bx����  �          A\)�G�@�����z���Bә��G�@��H���
�@G�B�Ǯ                                    Bx����  
�          AQ�ٙ�@ʏ\����=qB�\�ٙ�@�p��ۅ�X33B���                                    Bx���:  T          A�
��(�@����
=��B�𤿼(�@��R�ᙚ�V��B�\                                    Bx����  �          A
=q���@Ǯ���-��B����@�{��(��k�RB��                                    Bx���  
�          A{?�R@��
�Å�.B��H?�R@�Q����H�l=qB�W
                                    Bx���,  �          A�=#�
@�(���p��'
=B�B�=#�
@����
=�e  B�                                      Bx���  T          A�H=u@�R��{���B�=u@�����L��B��q                                    Bx��x  T          A�H����@���������B�uþ���@Ǯ��ff�;�
B��{                                    Bx��#  �          A�R�   @�����G��G�B�k��   @�������C=qB�k�                                    Bx��1�  �          A�Q�@�ff��p��
ffB��{�Q�@�G���\)�H�B�#�                                    Bx��@j  �          Ap��8Q�@�����{B���8Q�@�ff��
=�R�B�(�                                    Bx��O  �          A(��n{@�=q���'�BÅ�n{@����G��e33B�p�                                    Bx��]�  T          A�R�fff@�Q���z��9��B�G��fff@�  �G��w{B���                                    Bx��l\  
�          Azᾳ33@�33���
�=�B�zᾳ33@�33� Q��{\)B���                                    Bx��{  P          A�>�@�  ���H�P�RB�aH>�@X������{B�                                    Bx����  X          A��>u@�����C��B�>u@j�H���RB�ff                                    Bx���N  �          A
==�Q�@������
�<�B�L�=�Q�@~{��
=�{33B�z�                                    Bx����  
Z          Az�?J=q@�����=q�8�\B�
=?J=q@�33��
=�vffB�#�                                    Bx����  �          A
ff?@  @�p���{�Az�B��f?@  @n{��\)�Q�B�.                                    Bx���@  f          A{?��@��H����,�B�\?��@��R���i��B��
                                    Bx����  �          A�\?˅@�������(�B��?˅@�����  �e(�B�u�                                    Bx���  �          A��?��@�{��
=�"��B�� ?��@����  �_  B�G�                                    Bx���2  �          A
=@�R@��
���
�,�
B�
=@�R@�����fz�By�                                    Bx����  �          A�
@!�@�33����3  B�G�@!�@z�H����jBd�R                                    Bx��~  �          A�\@B�\@�Q��У��<  Bp{@B�\@S33��ff�o(�B=�                                    Bx��$  �          A��?�
=@ƸR��p��8Q�B�Ǯ?�
=@����t33B��                                    Bx��*�  �          Az�Ǯ@�z���
=��B�
=�Ǯ@�Q��ۅ�?ffB�{                                    Bx��9p  �          A��=��
@�\)���H��RB���=��
@�Q����P��B�k�                                    Bx��H  �          Aff�0��@�����ff���B�{�0��@�=q���
�Ez�B��H                                    Bx��V�  
�          A�
����@����
=�ffB�uý���@�
=����S�\B��                                    Bx��eb  �          A�H?�G�@��������=qB��f?�G�@������]��B�(�                                    Bx��t  �          A�\?ٙ�@׮��=q�){B��q?ٙ�@�Q�����e�B�Ǯ                                    Bx����  �          A�R@\)@θR��Q��1(�B���@\)@��
�(��j��BqQ�                                    Bx���T  �          A=q@+�@�  �ڏ\�4z�B��=@+�@����z��l�Bdz�                                    Bx����  �          A�H@��@ə���{�7z�B�G�@��@���ff�p��Bp�                                    Bx����  T          A�@-p�@�33�ҏ\�*��B�\)@-p�@���{�c�\Bm                                      Bx���F  �          A�@�=q@�z������ =qBW  @�=q@\)����O�HB*�
                                    Bx����  �          A�?��AG��333��ffB���?��@��R��p��33B�z�                                    Bx��ڒ  �          A�
@��Aff����c�B�Q�@��Aff������RB�\                                    Bx���8  �          A�\?�ffA�Ϳ�ff���B�{?�ffAQ�������B�aH                                    Bx����  �          A@@��@�
=�����Ez�Bt�@@��@QG��  �y��B=�                                    Bx���  �          A��@@  @���(��*��B|��@@  @�������a33BU�                                    Bx��*  
�          A=q@fff@�  �ۅ�5G�Bg@fff@i����R�g��B5z�                                    Bx��#�  �          A�R?�Q�@�  ��G��  B��
?�Q�@�\)���Y(�B��                                    Bx��2v  �          A!p����@�Q�@�=qA���B�
=���Aff@A?\)B�#�                                    Bx��A  �          A$  ���\@�=q@z=qA��\B��
���\A��?�
=A  B��=                                    Bx��O�  �          A������@��@tz�A�B��{����A�?�z�A��B�=q                                    Bx��^h  �          A*{��33@�z�@�G�A�G�C}q��33A�@HQ�A��HB�z�                                    Bx��m  �          A5p����@��
@�B
�C����@���@�  A�
=C�R                                    Bx��{�  �          A;
=��p�@��@�
=BC{��p�@�@��A�
=CE                                    Bx���Z  �          A:ff���@�@�{B�
Cu����@�ff@��A�\)C	=q                                    Bx���   �          A;\)��@�G�@�  B�C��@��
@�G�A�  C�
                                    Bx����  �          A=G��@u�@�z�B=qCh��@�
=@���A�
=C��                                    Bx���L  �          A<�����@���@�Q�B�HC�H���@�@ÅA��RC��                                    Bx����  �          A;���  @�Q�@�=qB'p�C����  @�Q�@��B�RC8R                                    Bx��Ә  �          A;\)��@fff@�G�B'33C����@��@ϮB
=C�R                                    Bx���>  �          A;�
���@H��A{B/ffC=q���@�Q�@�ffB=qC8R                                    Bx����  T          A:�\���
@p��@��B,��C����
@�=q@���B
�C��                                    Bx����  �          A:{��R@j=qAB1�Cٚ��R@�Q�@���B�CW
                                    Bx��0  �          A9p��33@h��@׮B�CG��33@��@�
=A�33CY�                                    Bx���  �          A9���
�\@|��@��
B33Cz��
�\@��
@���A�=qC                                      Bx��+|  �          A:�H���@��@�\)B�HC�=���@�=q@�Q�AᙚC��                                    Bx��:"  �          A:�R���@���@�G�B!�Cz����@�p�@��A���C                                    Bx��H�  �          A9����@��H@�  B(p�C����@љ�@�BG�C	ٚ                                    Bx��Wn  �          A-��
=@��@�G�B#  CE��
=@�p�@�\)A�=qCG�                                    Bx��f  �          A-�����@��@���B33C{���@��@���A��HC:�                                    Bx��t�  �          A-����@��H@�p�B��C���@�Q�@�
=A�z�B�p�                                    Bx���`  �          Aff��ff@�\@G
=A���B�����ff@��?u@�=qB���                                    Bx���  T          A�R���
@�(�@�Q�A���B������
A�R@&ffAzffB�G�                                    Bx����  �          A�R�ָR@�  @�A��C�3�ָR@�@Z�HA�G�C
��                                    Bx���R  �          Ap��θR@�{@�ffB
=C��θR@�33@�=qA�=qC\                                    Bx����  �          A!������@��@���B�C������@ڏ\@�A��
CO\                                    Bx��̞  �          A#\)����A��@?\)A�  B������AQ�>�G�@{B�q                                    Bx���D  �          A!����z�A Q�@l(�A��B���z�A�?�=q@�B�                                    Bx����  �          A�\���@�(�@��A�ffB��)���Ap�@A[�B�{                                    Bx����  �          A!����z�AQ�@^{A���B����z�A�\?�ff@�ffB�\                                    Bx��6  �          A"�H����A�@UA�G�B�.����A�?\(�@��HB�
=                                    Bx���  �          A!��(�A�@g
=A��B��f��(�Aff?�\)@�33B�\                                    Bx��$�  �          A!����  @�(�@�33A̸RB�u���  Az�?��RA6�RB�L�                                    Bx��3(  �          A"�H���@�{@a�A�z�B��R���A�?�G�@�(�B�(�                                    Bx��A�  T          A!��\)@ᙚ@�(�A��C���\)@��@  AN�\B�=q                                    Bx��Pt  �          A!�����@ڏ\@�z�A�
=C (����@�\)@C33A���B�Ǯ                                    Bx��_  �          A$(���G�@�  @�p�A͙�B�ff��G�A�H@
=qADQ�B�8R                                    Bx��m�  �          A#
=���\@�  @�=qA���C�����\@�
=@S�
A�\)B�                                    Bx��|f  �          A#����
@ٙ�@��A�ffC�=���
@�
=@Dz�A��B�\)                                    Bx���  �          A#33���R@�p�@�Q�A�C����R@�
=@*=qAr�RB�8R                                    Bx����  �          A#�
��33@�33@�Q�A�G�C����33@��\@�A\(�B��H                                    Bx���X  �          A#���Q�@��
@n{A��HC33��Q�A?��
A	�B�\                                    Bx����  �          A(z���(�@�p�@�{A�ffCW
��(�A(�@0��As�B��H                                    Bx��Ť  �          A)����H@�G�@��HA�\)C �����HA��@(Q�Ah  B��f                                    Bx���J  �          A)���z�@ᙚ@���A��C �f��z�A�R@7�A�
=B�=q                                    Bx����  �          A'����
@��\@ӅB�Cu����
@�{@���A��HC8R                                    Bx���  T          A'33��@�\@�G�A�G�B�
=��A
=@3�
Az{B��                                     Bx�� <  �          A'
=���@�Q�@�G�A�ffB��=���A
ff?�(�A.ffB���                                    Bx���  �          A((����R@�
=@�
=A��
B�
=���RA	G�?�A(z�B�L�                                    Bx���  �          A&ff���@�@�ffAʸRB�B����A�
@(�AB=qB��                                    Bx��,.  �          A&ff��(�@�ff@��B��C\��(�@��@a�A���B�\)                                    Bx��:�  �          A'\)���\@�(�@���B�C����\@�=q@��HA���B�33                                    Bx��Iz  �          A)�����@��@���B&�HC�R���@���@�Q�A�RC �=                                    Bx��X   �          A+���\)@�p�@��HBC���\)@��H@�  A޸RC(�                                    Bx��f�  �          A)G���Q�@�G�@�z�B"�HC
�H��Q�@߮@���A�C��                                    Bx��ul  �          A)����H@�G�@���B?�C� ���H@���@��B�\C��                                    Bx���  �          A'�
���@z�H@�\)BF{C�����@�\)@ҏ\Bp�C�3                                    Bx����  �          A)���
=@eA�BT�Cu���
=@���@���B*
=C�q                                    Bx���^  T          A(z���G�@K�A�Bc�\CxR��G�@��@��HB8��C)                                    Bx���  T          A!���H@���@�G�B�\CQ����H@�\)@\��A���C=q                                    Bx����  �          A(���=q@�(��z��^{C B���=q@�ff�����љ�C\                                    Bx���P  �          A\)�Å@���W
=���HCxR�Å@����2�\��33C�{                                    Bx����  �          A�\����@�����˅Cc�����@љ��>{��G�C�q                                    Bx���  �          A  ��G�@�녿�{�G�C����G�@��H�`  ��=qC�
                                    Bx���B  �          A
=��(�@��H?Q�@�\)C �R��(�@��ÿ��R���C �3                                    Bx���  �          A���G�@���?#�
@{�CY���G�@�녿����p�C��                                    Bx���  �          A����  @��R�$z��t��B�33��  @�{��p����B�z�                                    Bx��%4  �          AQ���p�A��=q�dQ�B�33��p�@����\��{B���                                    Bx��3�  �          A=q����A(��z��^�\B�����@陚��Q����B��                                    Bx��B�  �          A
=����A (��Q��d��B��H����@�G���  �޸RB��{                                    Bx��Q&  �          A�����@���Ǯ�%�C�\���@������fffCp�                                    Bx��_�  �          A�H���H@�  @I��A�Q�C�����H@�?�33A!p�CG�                                    Bx��nr  �          A
=�ff@q�@,(�A�p�C
�ff@�(�?��A�C�q                                    Bx��}  �          A33���@}p�@���A�
=C�3���@��H@XQ�A�C5�                                    Bx����  �          A����(�@���@�(�A�33C����(�@�  @J�HA�C�                                    Bx���d  �          Aff��p�@��@�\)A�RC33��p�@�33@>�RA���C
��                                    Bx���
  �          A�\���
@���@�
=B 
=Ck����
@�33@dz�A���C�\                                    Bx����  �          A{��p�@���@�A�C���p�@�33@!�Az�\CW
                                    Bx���V  �          A���=q@0��@���BCff��=q@�  @���B C�                                    Bx����  �          A����  ?��@��HBKz�C%}q��  @?\)@��B2p�C!H                                    Bx���  �          @�(�����?�  @�\)B1�\C �f����@G�@���B�HCL�                                    Bx���H  �          @Å��Ϳ�=q@�  Br�CK���>�=q@��B|��C-�R                                    Bx�� �  �          A���@���?n{@���C
�)���@��\����}p�C
Q�                                    Bx���  �          A33�љ�@�ff�\)�\(�Ck��љ�@�p���33�?33C��                                    Bx��:  �          A����
@љ��#�
��  CT{���
@�G����5C�                                     Bx��,�  �          A�\�љ�@�
=�.{��z�CE�љ�@����   �v{C:�                                    Bx��;�  �          Az��љ�@��
�Y����(�C���љ�@�(��,����C�                                    Bx��J,  �          A����Q�@�
=��{��(�C\��Q�@����>{��ffC��                                    Bx��X�  �          A�����@������ӅCs3����@�G��@  ����C�
                                    Bx��gx  �          Ap����H@��H��(��%�C�=���H@ָR�ff�d��Cff                                    Bx��v  �          A
=��ff@�p�?J=q@�{C�q��ff@��������
C8R                                    Bx����  �          A��ȣ�@���>\@��C=q�ȣ�@�\)��33��C�                                    Bx���j  �          A
=�Å@�����H���C���Å@�G��K�����C�R                                    Bx���  �          A!G��ȣ�@��G��8��C}q�ȣ�@أ�������(�C��                                    Bx����  �          A�
��  @�?�\)A5G�C.��  @�{���#�
C�                                    Bx���\  �          A(��У�@�  @%Av�HC��У�@�ff?&ff@vffC.                                    Bx���  �          A  ��33@ָR?�33@ۅC����33@أ׿8Q���Q�CY�                                    Bx��ܨ  �          A  �ҏ\@�G���R�j�HC{�ҏ\@�33� ���q�C��                                    Bx���N  �          A\)��Q�@��?(��@s33B�����Q�@����G���B���                                    Bx����  �          A Q���z�A�>u?�{B�=q��z�@��\���H�5p�B��H                                    Bx���  �          A��θR@�\)�z��Y��CǮ�θR@ٙ��!G��pz�C�                                    Bx��@  �          A  ���@�p���=q�˅CaH���@�=q�G��X  C                                    Bx��%�  �          A�\��@�{����J�HC ����@�  �'
=�uG�Cs3                                    Bx��4�  �          A�
��
=@��<#�
<�C�R��
=@����=G�C��                                    Bx��C2  �          A�
����@�\)?�33@�(�C0�����@��ÿG���  C��                                    Bx��Q�  �          A(���p�@�{@
=qAMC	���p�@���>�{?�p�C�
                                    Bx��`~  �          Ap��أ�@���?xQ�@�ffC��أ�@��ÿu��z�C��                                    Bx��o$  �          A(���{@��?�G�A�\C(���{@�=q��33�33Cz�                                    Bx��}�  T          Ap���  @߮?�33A
=C���  @��
=q�J=qC��                                    Bx���p  �          A=q��ff@�{?L��@���C�q��ff@�z῕��=qC0�                                    Bx���  �          A��p�@�G�?��
A&=qC����p�@�׽��0��C�{                                    Bx����  �          A�H��ff@�
=?�=q@�  C�)��ff@�  �h����  C�                                    Bx���b  �          A ����Q�@��R?�{@��
B�{��Q�@�
=�������
B�
=                                    Bx���  �          A33��\)Ap�?5@�=qB�����\)@�\)��  �
�\B�W
                                    Bx��ծ  �          Ap�����A=q<�>.{B�aH����@��H�
=q�J�RB�Q�                                    Bx���T  �          A\)��z�A �׿�  ��Q�B����z�@���W
=���HB��{                                    Bx����  �          A=q���@����
�\)C�
���@�
=��(��8z�C�                                    Bx���  �          Aff��{@�(�>�33@33B�\��{@���\�%G�B�p�                                    Bx��F  �          A�����RA(��}p����B�z����R@�{�J�H����B�33                                    Bx���  �          A
=���A�����H��B�����@���dz����HB��q                                    Bx��-�  �          A\)��ffA������B��=��ff@��Y����p�B�
=                                    Bx��<8  �          A �����A
=�8Q���(�B�\���@�ff�8Q�����B�ff                                    Bx��J�  �          A"�\���A	���aG����B�aH���A ���H����z�B���                                    Bx��Y�  �          A!����G�A=q��p���\B�k���G�@�(��z=q���\B�u�                                    Bx��h*  �          A���ff@�Q�#�
�mp�C �f��ff@陚�+��zffCT{                                    Bx��v�  �          A=q��G�@�z�z��XQ�C�=��G�@޸R�!G��l��C5�                                    Bx���v  �          A33��ff@񙚾�z��Q�C� ��ff@�ff��\�T��Cٚ                                    Bx���  �          A ����\)@�p����@  C8R��\)@أ��=q�^�\Cٚ                                    Bx����  �          A"{��p�@�녿p�����HCh���p�@�G��:=q��Cp�                                    Bx���h  �          A!���У�@��
�����
=C��У�@ᙚ�G
=��C�q                                    Bx���  �          A!���  A Q쿵� ��B�����  @��_\)��\)C0�                                    Bx��δ  �          A �����H@��H��G��
�\B��3���H@���a�����CaH                                    Bx���Z  �          A"�R���A�R�
=�?�
B�k����@�=q��{����B�aH                                    Bx���   �          A"{���A�Q��Yp�B������@�ff�����
B�W
                                    Bx����  �          A#���Q�A(��(���p  B����Q�@�Q���G��߮B�\)                                    Bx��	L  �          A!��33A=q�#33�i�B� ��33@����G���=qB�Ǯ                                    Bx���  �          A ����  @�=q��\)�ǮC����  @ٙ����2=qC�{                                    Bx��&�  �          A (��=q@�(�@B�\A�Q�Cs3�=q@���?��
A%G�C)                                    Bx��5>  �          A ���@��@U�A�
=C��@��?�Q�A2=qC�\                                    Bx��C�  �          A�
��z�@�=q@QG�A���C(���z�@��?�G�A$��C��                                    Bx��R�  �          A�
�5�A�H�P������B�.�5�@�����\)�=qB�\                                    Bx��a0  �          A"=q���
@�(���ff�33B�Q����
@޸R�_\)��G�B�W
                                    Bx��o�  �          A#\)�У�@����.{�z�HC�f�У�@�\)����H  C\                                    Bx��~|  �          A#���\)@�(���������C W
��\)@���H�����RCp�                                    Bx���"  �          A#33����A
=� ���d��B�������@�  ����ԣ�B��=                                    Bx����  �          A ������A �ÿ�Q����HB�L�����@�
=�O\)���B�p�                                    Bx���n  �          A"ff���Aff��Q��
=B�����@�  �`����Q�B��                                     Bx���  T          A=q��z�@�G���z���B�8R��z�@���XQ���Q�CxR                                    Bx��Ǻ  �          A=q��(�@�\�/\)��p�B�(���(�@Ӆ����߮Cc�                                    Bx���`  �          A
=����@��
�   �9��C ^�����@�33�y����
=Cff                                    Bx���  �          A!G��Ǯ@����
���C)�Ǯ@����]p����
C�H                                    Bx���  �          A#
=��p�@�ff�����(�C
�
��p�@����p��333C)                                    Bx��R  �          A"�\���
@�p���
=���C�����
@���z��S�
C8R                                    Bx���  �          A!��Ϯ@�\)�Ǯ��C��Ϯ@ڏ\�[���Q�C��                                    Bx���  �          A ����ff@�33�{�L��C\)��ff@�G��~{���RC��                                    Bx��.D  �          A$(���@�z�?�@ҏ\C5���@�p��c�
���RC{                                    Bx��<�  �          A#33��Q�@�{?���A{CG���Q�@��H�\�
�HC��                                    Bx��K�  �          A#�
���@��?�\)AffC\)���@�\)��Q���\C�                                     Bx��Z6  �          A$Q���33@���@�A;33C&f��33AG�<��
>�C 0�                                    Bx��h�  �          A#�
�ָR@�Q�?�z�A+�C��ָR@�  ��\)�\C�H                                    Bx��w�  �          A#���@�R?���A'
=C�
��@���G��!G�C                                      Bx���(  �          A#�����@�?�33A+\)Ch�����@�=��
>�G�C^�                                    Bx����  �          A"=q��\)@���?�A(  Cٚ��\)@��=u>���C�
                                    Bx���t  �          A"�H���@�?�z�A�\C����@�p��k����C��                                    Bx���  �          A"ff����@��@ ��A7
=C�q����@�ff>L��?���C�
                                    Bx����  �          A!���p�@�ff@��A[33Cٚ��p�@�\?\)@J�HCW
                                    Bx���f  �          A"ff�أ�@��@Q�AC�Cff�أ�@�R>��?�p�C5�                                    Bx���  �          A!���p�@���@#33Ak\)C��p�@�
=?c�
@��\C	�
                                    Bx���  �          A ����(�@��H?�p�A�CL���(�@�Q�#�
�k�C�\                                    Bx���X  �          A ����@�=q?���@�\)C33��@�
=����^�RC�                                     Bx��	�  �          A!G��@��@z�AT��C��@���?^�R@�{C                                      Bx���  �          A ����ff@��
?�(�A5�C���ff@�p�>\@	��C.                                    Bx��'J  �          A!���@��
@��Aa�C�3��@���?xQ�@���C��                                    Bx��5�  �          Ap���G�@�@ffAE��C�\��G�@ȣ�?\)@P  CB�                                    Bx��D�  �          A=q��{@�33@N{A�ffC
=��{@�\)?�(�A!�C
                                    Bx��S<  �          A�����
@�\)@hQ�A��HCT{���
@��@=qAdz�CO\                                    Bx��a�  �          Aff��Q�@���@�p�A��HC�)��Q�@��@R�\A�
=C@                                     Bx��p�  �          AQ����@}p�@n{A�
=C�f���@�  @'�A|  C@                                     Bx��.  �          A���ff@z=q@u�A�\)C\��ff@��@0  A��Cc�                                    Bx����  �          Ap�����u@��HB�C4}q���?���@�B�RC).                                    Bx���z  �          A����
=>�ff@�  B=qC0L���
=?�ff@�  A�=qC'�                                     Bx���   �          A
�\��Q��HQ�@��BG�CN�=��Q��z�@�{B�CD��                                    Bx����  �          A
=�����{�@�
=A���CT������-p�@�G�B�HCKY�                                    Bx���l  
�          A  ��G��1�@���BCK�)��G���z�@�G�B'��C@�H                                    Bx���  �          A����zῧ�@��HA�{C>
=��z�W
=@���A��C5�H                                    Bx���  �          A=q��33�Q�@���A�z�C9����33>B�\@��A�
=C2�H                                    Bx���^  �          A���@=p�@�\)A�ffC^���@z=q@X��A�G�C8R                                    Bx��  �          A�H���H@�G�@U�A���C�
���H@�
=@(�Ab�RC�q                                    Bx���  �          A�����
?���@��
B6�C)�����
@!G�@�B&�\C�=                                    Bx�� P  �          A����R?B�\@��BG��C,� ���R@{@أ�B9ffC��                                    Bx��.�  �          A  ����?@  @�=qB<�C-8R����@�@�B/{CQ�                                    Bx��=�  �          A��{@.{@S33A�  C ����{@\(�@"�\A�ffC�                                    Bx��LB  �          A\)��(�?�z�A�B^��C'�
��(�@Dz�@�BJ��CQ�                                    Bx��Z�  �          A!�����@��ABVG�C�����@�Q�@陚B9��C+�                                    Bx��i�  �          A!���
=@
=@�p�B6(�C"���
=@qG�@θRB�C�                                    Bx��x4  �          A���@HQ�@�33AŮCz����@���@O\)A�  CQ�                                    Bx����  
�          Az���@K�?�\)A:�HCff��@b�\?��@�=qCT{                                    Bx����  �          A�
�G�@N�R?�@XQ�C�=�G�@QG������33C��                                    Bx���&  �          A���G�@�\)?�{A%G�C���G�@��?\)@eC�                                    Bx����  �          AG���ff@���?У�A*�\Ck���ff@�G�?!G�@��
Cٚ                                    Bx���r  �          A33��{@���?�{A�
C�R��{@�  >���@   C�{                                    Bx���  �          A�����@�?��RA
=C  ����@�33>aG�?�Q�C�q                                    Bx��޾  �          A�����R@�  ?��A(�C�
���R@�ff>�Q�@Ch�                                    Bx���d  �          A�����H@q�?�(�@�\)CB����H@}p�>���?�Q�C33                                    Bx���
  �          A�
��(�@N{?�
=AL  CǮ��(�@e�?�\)@�33C�\                                    Bx��
�  �          A�Q�@C�
@  Ad��C�3�Q�@`  ?�(�A��C\                                    Bx��V  �          A\)�	@R�\@ALz�C��	@k�?��\@�{C�{                                    Bx��'�  �          A  � Q�@c�
?�p�AK�
C\� Q�@{�?���@��C�                                    Bx��6�  �          A����@s33?�z�AD��CO\���@�z�?z�H@�  CW
                                    Bx��EH  �          A
=��@E�?�\)A,Q�C�=��@W�?\(�@�ffC��                                    Bx��S�  �          AQ���p�@3�
?�ffA��C u���p�@B�\?�R@���C�                                    Bx��b�  �          A	��� ��@)��?�(�A33C!� � ��@7
=?�@vffC h�                                    Bx��q:  �          A�H�p�@xQ�?�A)�Cc��p�@��?=p�@��
C��                                    Bx���  �          A33���@�33?�G�@�(�C!H���@���>�\)?�  C!H                                    Bx����  �          A
=���\@��R?u@�z�C�����\@�=q���aG�C!H                                    Bx���,  �          Az��ff@���?�ff@��HC��ff@����\)��ffC�                                    Bx����  �          A
=��
=@�?^�R@��\C=q��
=@�  ��=q��C�H                                    Bx���x  �          A�R���H@�{?�=q@��HC\)���H@����B�\��33Cٚ                                    Bx���  �          A����\)@�Q�?���@�ffC	�q��\)@��H����Q�C	�f                                    Bx����  �          Ap����R@�=#�
>��C�����R@�G����R��RCL�                                    Bx���j  �          A33���@�G�=���?(�C=q���@����G���G�C�)                                    Bx���  �          A\)��R@��>#�
?z�HCxR��R@�G���p���G�C�                                    Bx���  W          A33��@��H?�ff@ə�C
B���@�p��������C	�                                    Bx��\  �          Ap��z�@��?��
A�
C��z�@��R>�p�@�Cٚ                                    Bx��!  �          A��Q�@��?@  @�33CG��Q�@��H���*�HC�                                    Bx��/�  �          A�\��
@�ff>�ff@%�C}q��
@�p��G���G�C�                                    Bx��>N  �          A�����@�(�������
C
#�����@�(���\)�,��CB�                                    Bx��L�  �          A�\��(�@�(���z���Q�CB���(�@�
=�(��dQ�C33                                    Bx��[�  �          A�G�@��׾�  ��p�C���G�@��\��z��  C�)                                    Bx��j@  �          Aff��@��\���0��C���@�33�˅��C5�                                    Bx��x�  �          A�����@��ü#�
�uC����@�������\)CaH                                    Bx����  �          A����@�z�>�z�?��HC����@�33��R�i��C+�                                    Bx���2  �          A{��@���?��H@�=qCT{��@��<�>#�
C��                                    Bx����  T          A=q���@�  @�\A>�HC\���@�=q?W
=@��C�                                     Bx���~  �          A
=�\)@�33?�33Ap�C���\)@��\>�ff@%C�                                    Bx���$  �          A33����@��H?��RA	�C������@���>B�\?���C��                                    Bx����  �          A�R��z�@�G�?�G�AQ�C�R��z�@�\)>aG�?��\C�
                                    Bx���p  �          A�\���@���@�\A>�\C����@��?�G�@�(�CY�                                    Bx���  �          A����@��R?���A�
C�f��@��>���@�C޸                                    Bx����  �          A=q�{@�
=@p�AO
=C���{@��H?��@��C�                                     Bx��b  �          A�\�	�@���?�p�A ��C� �	�@��?0��@���C(�                                    Bx��  �          A Q���@��\?h��@��C�)��@��;�����p�C�=                                    Bx��(�  �          A   �Q�@��H?Q�@�ffC+��Q�@�zᾔz���C�H                                    Bx��7T  �          A\)�	��@��R?�@Dz�C�	��@��R���Dz�C                                    Bx��E�  �          A���G�@�=q?��AG�CE��G�@���>�p�@
�HCB�                                    Bx��T�  T          A���{@��?�Q�A5p�C�3��{@�z�?(��@u�Ch�                                    Bx��cF  �          A=q���R@�
=?u@��HCQ����R@�G���=q���C��                                    Bx��q�  �          A z�����@���?��RA7\)C������@�=q?:�H@�Cz�                                    Bx����  �          A\)�   @��H?0��@~�RC�
�   @������J�HC�                                    Bx���8  �          A\)�z�@��\?�{@�z�Cz��z�@�Q�>��
?�{C�{                                    Bx����  �          A�H���@�G�?�
=Az�C�����@��R>B�\?�{C                                    Bx����  �          A$��� ��@���?��A!�C�� ��@�G�?   @1�C�                                    Bx���*  �          A"�\���@�\)?���@�Q�C�����@���>W
=?���CǮ                                    Bx����  �          A z��\)@�\)>Ǯ@p�Cs3�\)@��R�
=q�G
=C��                                    Bx���v  �          A"�H���@���?���@���C0����@��<��
=�Q�C��                                    Bx���  �          A$(���\)@أ�@l��A�\)C����\)@�z�@�RAIp�CW
                                    Bx����  �          A#���z�@љ�@qG�A�
=Cp���z�@�@ffAT��C��                                    Bx��h  �          A!���ۅ@˅@r�\A��\C	+��ۅ@�  @�HA]�Cn                                    Bx��  
�          A�
�߮@���@qG�A���C@ �߮@��@{Ae�CaH                                    Bx��!�  W          A!�����@�  @xQ�A�(�C@ ����@�@2�\A�G�C�
                                    Bx��0Z  �          A ���(�@X��@��\A�(�C@ �(�@���@S�
A�ffC�                                    Bx��?   �          A�� z�@�{@C�
A��CaH� z�@�ff@ ��A:�HC��                                    Bx��M�  �          A{��z�@�z�?(�@dz�C@ ��z�@�(��!G��j�HCB�                                    Bx��\L  �          A�\���@�@.�RA��HC:����@��
?�33AQ�C�                                    Bx��j�  �          A��\)@�(�?���A	G�C  ��\)@�=q>�33@�C�                                    Bx��y�  �          A��{@�G�@333A��RC�)��{@��?�  A$(�CT{                                    Bx���>  �          A�
��@���(��{C5���@�\)�˅��C&f                                    Bx����  �          A\)� ��@�(���ff�)��C�� ��@���ff�G�C�                                    Bx����  �          A��(�@�p�?@  @�33C���(�@�ff���1�C��                                    Bx���0  �          A����R@���?�\A&�RC�R��R@ȣ�?��@N�RC�                                    Bx����  �          A�����@�33?��HA	��C
޸���@�Q�>L��?�33C
&f                                    Bx���|  T          A������@���?���@��Cff����@�p�>L��?�C��                                    Bx���"  �          A  ����@��@p�AR{Cn����@�=q?�z�@ڏ\C�\                                    Bx����  �          A���ff@�33?��A2{C���ff@��
?^�R@��
C��                                    Bx���n  �          A����\)@��?�33A3
=C
��\)@�(�?W
=@���C��                                    Bx��  �          A�
� ��@��?�{A0  CQ�� ��@�p�?W
=@�C                                    Bx���  �          A=q��33@��
@(�AR�\C޸��33@�ff?�@�Q�C=q                                    Bx��)`  �          A�\�z�@���@ ��AAC�z�@�ff?��@�=qCh�                                    Bx��8  �          A{�@E�@`  A�p�C��@l(�@6ffA���C33                                    Bx��F�  �          A���\)@8Q�@]p�A��C!8R�\)@^�R@6ffA�z�C��                                    Bx��UR  �          A�\�
�R@;�@>�RA�=qC!L��
�R@\(�@Q�Ad��CY�                                    Bx��c�  �          A�����@���@�Af=qC����@��?У�AC��                                    Bx��r�  �          AG��Q�@�Q�?���A1��C���Q�@���?���@�z�CO\                                    Bx���D  �          A��
{@�{?У�A{C)�
{@�p�?Q�@�33C��                                    Bx����  
�          Az��
�R@o\)?��
@�\)C���
�R@z=q?�@Z=qC�R                                    Bx����  �          A���G�@�33?�\)@ҏ\C��G�@�\)>�33@�\CaH                                    Bx���6  �          A����@��H>�z�?ٙ�CB���@�=q���.�RCY�                                    Bx����  �          A33�
�\@�(��aG����
C+��
�\@��׿�G���  C�                                    Bx��ʂ  �          A���\@z=q=u>���CL���\@w
=�(���w�C��                                    Bx���(  �          A{��R@�ff��
=�{CǮ��R@�����
=�ۅC�\                                    Bx����  �          A\)�
ff@��=��
>�(�Ch��
ff@��H�O\)��{C�                                     Bx���t  �          A (����@�  ��G��#�
C�
���@��ͿxQ���G�C
                                    Bx��  �          A=q�
=q@�ff��33��Cz��
=q@�녿�
=��p�C0�                                    Bx���  �          Az��33@w���G��%C���33@n�R��\)���
C\)                                    Bx��"f  �          A=q�
�R@��������
=C�R�
�R@s33��  �'33CW
                                    Bx��1  �          A33���@vff�O\)���HCaH���@i����p��p�C��                                    Bx��?�  �          Aff�(�@���?�G�A�CG��(�@�\)?:�H@�=qC:�                                    Bx��NX  �          A���R@�p�?��
@ǮC����R@�G�>�\)?�Q�C�                                    Bx��\�  �          Ap��
=q@\)?��@ǮC0��
=q@��
>���?���C��                                    Bx��k�  �          A{���@���?5@��\C�����@��\�L�;�\)C��                                    Bx��zJ  �          Ap�����@�{?ǮAC������@�z�?.{@��C�                                    Bx����  �          A�R��
=@���?�@�
=CQ���
=@�z�>#�
?uC                                    Bx����  �          Aff�љ�@��H?�\)A�C�\�љ�@�\)>W
=?��C8R                                    Bx���<  �          Az���Q�@�zῘQ���C{��Q�@ڏ\����k�CT{                                    Bx����  �          A����\@�{>�{@B����\@�z�n{��  B�#�                                    Bx��È  �          A�����\@�R>L��?�  B������\@�(�������B�.                                    Bx���.  �          A�\��\)@�
=?�
=@�
=B�8R��\)@�녾\)�Y��B���                                    Bx����  �          A  ��Q�@�z�?���A4  B����Q�@�33?�@O\)B��R                                    Bx���z  �          A�����\@�{@8��A��B�B����\@�=q?���A��B���                                    Bx���   �          A�\����@�\)?&ff@y��C ޸����@�����e�C ٚ                                    Bx���  �          A�R��Q�@��?G�@��C�{��Q�@��H��ff�)��Cz�                                    Bx��l  �          A�H�ʏ\@陚?(��@z�HC��ʏ\@陚����QG�C��                                    Bx��*  �          A33��{@�ff>�(�@!�C��{@���L�����C޸                                    Bx��8�  "          A�R��(�@�z�?z�@^{B�#���(�@�(��+��\)B�.                                    Bx��G^  �          A\)��\)@�(�>�=q?�=qC {��\)@�=q�z�H��G�C L�                                    Bx��V  �          A�
��33A�\?!G�@n�RB����33Aff�=p���33B���                                    Bx��d�  �          A������A�
?L��@�ffB������A(��z��X��B�\                                    Bx��sP  �          A����(�A=q��G��%�B�����(�@��R�޸R�$(�B�#�                                    Bx����  �          A����@�ff@ffAV�HCQ���@�ff?�  @��
CJ=                                    Bx����  �          A�\���
@��\@Q�A]��C	�=���
@�33?�
=@��HCE                                    Bx���B  �          A=q��
=@ʏ\?��A�
C����
=@Ϯ?�@Mp�C                                      Bx����  �          A#\)��Q�@�?.{@uC���Q�@�Q����*�HC��                                    Bx����  �          A%G���33@�?^�R@��HC����33@�\)������Q�C�                                    Bx���4  �          A%���\)@��>�Q�@   C����\)@��ÿ@  ����CǮ                                    Bx����  �          A"=q��33@�=q��(���ffC	&f��33@љ�����O\)C
O\                                    Bx���  �          A�\����@����H�(��C�����@��H�0�����CxR                                    Bx���&  �          A����@�p���G33C Q����@�Q��L����z�C��                                    Bx���  �          A����AG��
=q�J=qB������@���Y������B�\                                    Bx��r  �          A����
@�\)�ff�D��B��f���
@�=q�R�\���B��q                                    Bx��#  �          Az��u�A���H�	�B�Ǯ�u�A���2�\����B�\)                                    Bx��1�  �          A�\�\(�A{���W\)B�W
�\(�A
=�e��  B�W
                                    Bx��@d  �          A!�����HA�R�C33����Bˀ ���HA����{B��                                    Bx��O
  �          A{���\Aff�(Q��w�B�\)���\Aff�\)����B�33                                    Bx��]�  T          Ap��  Az��L(����B���  A\)�����
=Bѽq                                    Bx��lV  �          A��3�
A	��dz����HB�W
�3�
@�ff������
=B���                                    Bx��z�  �          A��{A
�H�fff��  BӸR�{A �����\���B�
=                                    Bx����  �          A\)�\)A��,(���
=B�  �\)A��~{��p�BԮ                                    Bx���H  �          A���A���ff���B��f�A���&ff�\)B���                                    Bx����  �          A��-p�A�R���
���Bמ��-p�A�H��R�hz�Bؔ{                                    Bx����  �          A
=���RA�Ϳ�(��LQ�Bǳ3���RA�H�J�H��
=BȔ{                                    Bx���:  �          A��<��
A������υB�Ǯ<��
@��
���\��B�                                    Bx����  �          A����=qA����H��{B½q��=q@�Q���
=�Q�B�L�                                    Bx���  �          Aff����A33��\)��  B�#׿���@�R���H��RB�G�                                    Bx���,  �          Az῝p�A  �w
=�Ə\B��῝p�@�(���\)��B�W
                                    Bx����  �          Ap��(�A��e����B�\�(�@�����{��Q�B�ff                                    Bx��x  �          A=q�!�A33�=p���(�B�\)�!�@��R�������B�aH                                    Bx��  �          A  �33A��(��c
=Bҽq�33@���U���z�B��                                    Bx��*�  �          A�
��A�Ϳ��\�{Bӽq��A�����{
=BԨ�                                    Bx��9j  �          A
=��A=q����B�HB�{��A ���@������B�L�                                    Bx��H  T          A  ����A
=q�����:�HBȔ{����A���>{����B�aH                                    Bx��V�  �          A���
�HA�Ϳ�(��I�B�p��
�HA\)�Fff��ffBє{                                    Bx��e\  �          A���"�\AQ쿢�\�=qB�=q�"�\Az�����w�
B�.                                    Bx��t  �          A���G�A
=q������z�B�aH�G�A�\��pQ�B�(�                                    Bx����  �          A{��
A
�\��ff��B����
A{�+����B��H                                    Bx���N  �          Ap��%A
=q�����B�aH�%A�H���^=qB�.                                    Bx����  �          A녿��A(���Q���\B�
=���A  �$z���  B˽q                                    Bx����  �          A�
���RA33�{����B�  ���RA ���c33��G�B���                                    Bx���@  �          A��L��@�G���Q���=qB��3�L��@���\)��\B��                                    Bx����  Q          A���A��
=�`��B����@�z��H����B��)                                    Bx��ڌ  W          A�H�#�
A���33�p��B��\�#�
A�H�W�����B���                                    Bx���2  �          AQ�?�33AG�@-p�A�{B��f?�33A	?У�A)��B��                                     Bx����  �          A�?�Q�A�H?�\)AD��B�B�?�Q�A	��?L��@�
=B��3                                    Bx��~  �          A33?�(�A   @i��A�33B�#�?�(�A=q@'�A���B�{                                    Bx��$  �          A�?��@�
=@�  B(�B�G�?��@�Q�@z=qA���B�B�                                    Bx��#�  �          A���Q�A�H@33Av{B�aH��Q�Aff?��\A33B�Ǯ                                    Bx��2p  �          A�����@����\)��ffB�aH����@�33�xQ��ϮB���                                    Bx��A  �          A�\�У�@���-p����
C
E�У�@�{�X����Q�C&f                                    Bx��O�  �          A33��@��\�]p���33C����@��
��z���z�C
��                                    Bx��^b  �          A33���
@��H�|����{C#����
@�=q���
��{C	�=                                    Bx��m  �          Aff�θR@����_\)��(�C
�θR@�ff�����ffC�=                                    Bx��{�  T          A��ڏ\@�ff��H�w33Cp��ڏ\@��
�B�\���HC#�                                    Bx���T  �          AG���G�@��R���;�C���G�@��R�   ��(�C8R                                    Bx����  �          A���@�\)�����  Cff��@�����\�O�Cff                                    Bx����  �          A�H��(�@�  �G��M��C����(�@�\)�'
=��C
                                    Bx���F  �          A�H��R@�����Q��D(�C����R@�G��"�\���HC�                                    Bx����  �          A{��  @�ff����@��C�f��  @��R�=q�w�CG�                                    Bx��Ӓ  �          A���
=@������k33CB���
=@�Q��7
=���C��                                    Bx���8  �          A��ָR@�\)��z����HC
E�ָR@�녿����9C{                                    Bx����  �          A33��{@��
����ָRCu���{@��R��(��1��C	0�                                    Bx����  �          A  ��{@��Ϳ�p����
CW
��{@�
=��z��C�
C	(�                                    Bx��*  �          Az���z�@��Ϳ�\�N�RC!H��z�@�녿����ڏ\C�
                                    Bx���  �          A���z�@\�p����\)C	����z�@�{�����#33C
+�                                    Bx��+v  �          A�����H@�z���Q�CǮ���H@ə����H��{C5�                                    Bx��:  
�          A����=q@�z�=��
?�C���=q@Å�z��l(�C	�                                    Bx��H�  �          Ap���p�@��\�@  ��{C	� ��p�@�
=��\)���C
O\                                    Bx��Wh  �          A	����H@��?.{@�=qC	ٚ���H@���=�\)>��C	�f                                    Bx��f  �          A	���\)@�{��z����C
��\)@�����33�3
=C�                                    Bx��t�  �          A�����@���?��A�C@ ���@��?L��@�(�C�H                                    Bx�Z  �          A���(�@�(�?333@���CJ=��(�@�p�>#�
?���C�                                    Bx�   �          Az����@�
=�^�R���C#����@��������\CǮ                                    Bx� �  �          A����{@�녽�G��+�C����{@��ÿ
=q�\��C��                                    Bx�¯L  �          Aff���H@�Q��.�R���
C�����H@��R�J�H��(�CT{                                    Bx�½�  �          A{��@���\)�}p�C)��@~{�9����ffC�                                    Bx��̘  �          A���G�@"�\�z��n�HC#\�G�@�\�#�
��(�C$�)                                    Bx���>  �          A����@
�H�=q�|��C%B���?��'
=��p�C&��                                    Bx����  �          A�\�(�@ff�   ���RC%���(�?����,(����C'aH                                    Bx����  T          A	�� ��>Ǯ�A���ffC1B�� ��=�Q��C33��C3Y�                                    Bx��0  �          A�
�ff>L���?\)���HC2�{�ff��Q��?\)��33C4�)                                    Bx���  �          A��Q�?����\�k�C'� �Q�?�z��p��}p�C(�R                                    Bx��$|  �          AG���R?�=q������
C-
��R?z�H�����C-�q                                    Bx��3"  �          Az����?�ff�
=�r�HC)�3���?�=q�\)���\C+.                                    Bx��A�  �          A��@�0  ��C%�q�?����<(���p�C'��                                    Bx��Pn  �          A{��@1G��.�R���C!ff��@ ���>{��z�C#\                                    Bx��_  �          A��=q?���L(���
=C&��=q?˅�Vff���
C(�                                    Bx��m�  �          A�\�z�@33����k33C$s3�z�@�����=qC%�{                                    Bx��|`  �          A�R�(�?�p��.�R���\C&�=�(�?�(��9������C(5�                                    Bx�Ë  �          A=q���\?z������p�C/����\>#�
��{��\)C2�\                                    Bx�Ù�  �          Az��陚@��'���ffC$��陚?��2�\���HC%޸                                    Bx�èR  �          A	�����@j=q�aG����HC�f���@hQ�
=q�eCٚ                                    Bx�ö�  T          A(���
=@`�׾�=q��C���
=@^�R���w
=C�                                     Bx��Ş  �          A����@{���G��@  C����@y����ff�@��C��                                    Bx���D  �          AQ���G�@w�>���@{C�)��G�@xQ�        C�                                    Bx����  �          A  ��R@�Q�>8Q�?��HC����R@�Q�#�
����C��                                    Bx���  �          A(���=q@X�ÿ�p��\)C�{��=q@Q녿�G��&{C�\                                    Bx�� 6  �          A	����=q@���=L��>�Q�C�R��=q@�Q쾙���G�CǮ                                    Bx���  �          A\)���@���?�G�@��CW
���@�33?
=@z�HC�q                                    Bx���  �          A����@��?xQ�@�z�C0����@��
?\)@j�HC�)                                    Bx��,(  �          A  ��(�@�p�?�@�
=Cc���(�@�  ?G�@�33C�R                                    Bx��:�  �          A���\@�z�?Y��@��C���\@�{>�@J�HCaH                                    Bx��It  �          A���Q�@��\?8Q�@��C���Q�@��
>\@\)C�\                                    Bx��X  �          A\)��R@��
?�ffA#�
CW
��R@�\)?�
=@�=qC�q                                    Bx��f�  
�          A�
��\@��?�Q�A�CG���\@�  ?��@�ffC                                    Bx��uf  �          A  �ڏ\@��?��A(�C#��ڏ\@�z�?c�
@�=qC��                                    Bx�Ą  �          A����@�{?�{A=qCc����@���?s33@�{C�                                    Bx�Ē�  �          A�����@��H?�ffA"=qC�f���@�{?�33@�  CaH                                    Bx�ġX  �          Aff���
@���?�{Az�C
�H���
@��?n{@�  C
xR                                    Bx�į�  �          A����Q�@�
=?�@Tz�C
����Q�@��=�Q�?\)C
xR                                    Bx�ľ�  �          A(���p�@���>\)?fffC����p�@�zᾣ�
�z�C��                                    Bx���J  �          A33����@���?�A�\C�����@�\)?s33@\C�f                                    Bx����  �          A\)��@�33?�\)AAG�C ff��@޸R?���A�\B��H                                    Bx���  �          A�R��Q�@��?E�@�Q�C	Ǯ��Q�@�ff>�Q�@z�C	�
                                    Bx���<  T          A�R�ڏ\@��?Y��@�\)C\)�ڏ\@�ff>�@=p�C#�                                    Bx���  T          Ap���G�@��?���A�CaH��G�@\?��@���C��                                    Bx���  �          A���
@�p�?�\)A)�C	
���
@���?�(�@�{C�H                                    Bx��%.  �          Ap��ҏ\@�{?�G�AC&f�ҏ\@���?���@�C
�R                                    Bx��3�  �          A
=��
=@�?���A%��C+���
=@ȣ�?�Q�@�ffC�                                     Bx��Bz  �          A�H����@��
?У�A(��Cc�����@θR?�(�@��HC�R                                    Bx��Q   �          A�\���@��\?���A"ffC
^����@�p�?���@��RC	�                                    Bx��_�  �          A����@�\)?��HAL��C޸���@��H?ǮA"�HC\)                                    Bx��nl  �          A�����@�z�?�  A7�C� ����@�\)?�=qA\)CW
                                    Bx��}  �          A  ��33@��?��
A�
C	�\��33@�(�?k�@�33C	:�                                    Bx�ŋ�  �          A�
����@�  ?�ffA>�HC�����@�33?�
=A\)C��                                    Bx�Ś^  �          A=q���
@�\@E�A�z�B�\)���
@�  @)��A���B�#�                                    Bx�ũ  T          A�H����@�=q@(Q�A�\)B�������@�R@p�Ae��B��                                    Bx�ŷ�  �          A33��(�@�ff@$z�A��C���(�@��H@(�AbffC��                                    Bx���P  �          A{���H@�33@N�RA��B������H@���@5�A��B�Q�                                    Bx����  �          A��|(�@�@hQ�A�G�B�z��|(�@��
@N{A�Q�B�33                                    Bx���  �          A��j�H@��H@w�A�{B�33�j�H@�G�@\��A�
=B��                                    Bx���B  �          Aff�Z=q@�ff@aG�A���B�Ǯ�Z=q@�(�@EA�ffB���                                    Bx�� �  �          A���<(�@��@?\)A��HB�p��<(�A=q@#33A�\)B�                                    Bx���  �          A���?\)A Q�@4z�A�  B��H�?\)A�\@��AuB�=q                                    Bx��4  �          A  �L��A   @'
=A��HBߏ\�L��A�@�A`��B��                                    Bx��,�  �          AG��S�
A�@{A}G�B��{�S�
A
=@33AP��B�                                      Bx��;�  �          A���[�@�z�@7�A��B����[�A Q�@p�A|  B�B�                                    Bx��J&  �          A���G
=@�\)@.�RA���Bފ=�G
=A@z�Ao�
B��                                    Bx��X�  �          AQ��^�R@��@&ffA�{B��^�RA z�@��Ab=qB��H                                    Bx��gr  �          A  �8��@�@l��A�{B�G��8��@�33@Tz�A�
=B�p�                                    Bx��v  �          AQ��<��@���@FffA�G�B�  �<��A ��@-p�A�z�B�W
                                    Bx�Ƅ�  S          Az��z=q@�@(��A�  B����z=q@���@G�Ai�B�L�                                    Bx�Ɠd  �          AQ����@�@.{A���B� ���@�p�@�At  B�                                    Bx�Ƣ
  �          AQ���=q@�\)@�
An{B�\)��=q@�=q?�(�AJ�HB���                                    Bx�ư�  �          A�
��ff@ۅ@!�A��B�L���ff@�
=@{Ae�B�u�                                    Bx�ƿV  �          A����G�@ָR@(�Ab=qC����G�@ٙ�?��AB�RC0�                                    Bx����  �          AG��ƸR@ҏ\>��R?��RCL��ƸR@��H<�>.{CE                                    Bx��ܢ  �          A�����@׮=�G�?333C�3����@�\)�8Q쿐��C�3                                    Bx���H  �          A����Q�@У�>�{@
=qC�{��Q�@���=�\)>�
=C�=                                    Bx����  �          A����z�@��>u?�G�Cٚ��z�@�=q�#�
��\)C�{                                    Bx���  �          A(���p�@�p�?O\)@��C�)��p�@�{?�@Z�HC}q                                    Bx��:  �          A  ���H@ٙ���\)��G�C����H@��ÿ���`  C�                                     Bx��%�  �          A=q��
=@�Q�>���?�
=C:���
=@أ�<�>aG�C33                                    Bx��4�  �          A�R��G�@׮�W
=����C����G�@�
=����C33C�R                                    Bx��C,  
�          A
=�W
=@���@�A|��B�.�W
=@�@33A]G�B�3                                    Bx��Q�  �          A�\��33@�(�@�Aw�B�(���33@޸R@�A\Q�B��                                     Bx��`x  �          A�R��  @��H@�AqG�B�Q���  @�p�@�AVffB��3                                    Bx��o  �          A����@��@Dz�A��\B�G����@��@3�
A�ffB�{                                    Bx��}�  �          A�\����@�@l��A��HB�������@��@^�RA�(�B���                                    Bx�ǌj  �          Aff���@���@  Ak�
B�u����@�\)@G�AR�RB��f                                    Bx�Ǜ  �          A��g
=@�@���A�B�{�g
=@��
@q�A�{B�G�                                    Bx�ǩ�  �          Ap��6ff@��@�Q�A�B�\)�6ff@��@���A��Bޏ\                                    Bx�Ǹ\  �          A��S33@�\)@�ffB#{B��f�S33@��@���BB�ff                                    Bx���  �          A��QG�@��@��B=qB왚�QG�@�\)@���B��B�L�                                    Bx��ը  �          A
=�i��Ap�?�\@P��B�\�i��A>��?�
=B�                                     Bx���N  �          A��<(�A�ÿ=p�����B���<(�A�׿}p���z�B�
=                                    Bx����  
�          A(����A�H��
�Z=qB�\���A���\�s�
B�G�                                    Bx���  �          A
�R�I��@�{�h�����B�L��I��@���������B�p�                                    Bx��@  �          A	G���=q@��>u?�z�B�33��=q@��=L��>��
B�(�                                    Bx���  �          A����  @�=q@��A���B�����  @�(�@{Au�B�\)                                    Bx��-�  �          A33�Z=q@��
<��
>��B�33�Z=q@��8Q쿚�HB�33                                    Bx��<2  �          A
=�Å@��
@,��A��C��Å@�{@$z�A�{C��                                    Bx��J�  �          A����z�@�{@
=qAi��C�H��z�@��@33A^{CQ�                                    Bx��Y~  �          A
=q���H@p��@!G�A��
Cٚ���H@tz�@�A���C}q                                    Bx��h$  �          A
�R���@�p�@4z�A�Q�CB����@��@.{A��HC�H                                    Bx��v�  �          A	G��ƸR@�z�?�ffA'�
C	� �ƸR@�p�?�
=A{C	�{                                    Bx�ȅp  �          A
=q����@�33@�AzffC�=����@���@��Ak�C�\                                    Bx�Ȕ  �          A
{���H@�  @{Ao�
B�G����H@�G�@�A`(�B��                                    Bx�Ȣ�  �          A	p���p�@�33?�Q�AQ�B��)��p�@�z�?�ffAAp�B�{                                    Bx�ȱb  �          A	��c�
@�  ?�  AB�\)�c�
@���?���@��HB�=q                                    Bx���  �          A
ff�S�
@�z�?�  A��B��S�
@��?���@陚B�ff                                    Bx��ή  �          A
ff�C�
A (�?��@��B��)�C�
A z�?h��@���B�Ǯ                                    Bx���T  �          A
�\�!�A33?��@�{B�=q�!�A�?fff@�\)B�(�                                    Bx����  �          A
�H� ��A�
?.{@�\)B��� ��A  ?
=q@c�
B��H                                    Bx����  �          A	����ff@�\)?�=qAEp�B�B���ff@�Q�?��HA8��B�\                                    Bx��	F  �          A
=q���@��?ǮA'33B����@�?���A�HB��H                                    Bx���  �          A
�H��{@�Q�?��A0  B��{��{@���?��A$��B�ff                                    Bx��&�  �          A  �[�@�R@!�A��\B����[�@�  @�A��RB晚                                    Bx��58  �          A���X��@�=q@ ��AY��B�33�X��@��H?�33ANffB�\                                    Bx��C�  �          A	��\(�@陚@��A���B��\(�@�\@ffA��B�W
                                    Bx��R�  �          A�Ϳ\@��H@�  B�B�\�\@�z�@�p�B  B��)                                    Bx��a*  �          A
=�/\)@�=q@��RB G�B�W
�/\)@��
@�z�A�B�
=                                    Bx��o�  �          Az��Y��@���@�  B �RB����Y��@��@�{B��B�p�                                    Bx��~v  �          A	��E@ə�@�=qB

=B�=q�E@��H@�Q�B��B��f                                    Bx�ɍ  �          A
ff�1�@�z�@��B�HB�
=�1�@�{@���B�B�R                                    Bx�ɛ�  �          A	��   @�G�@��
BffB܀ �   @ڏ\@�=qB �B�G�                                    Bx�ɪh  �          A\)� ��@�R@XQ�A�{B�(�� ��@�@Tz�A��\B�                                    Bx�ɹ  �          A(��z�H@陚@#33A�
=B�ff�z�H@�=q@   A�  B�B�                                    Bx��Ǵ  �          A��j=q@�=q@(�AiB�\�j=q@�\@��AdQ�B�u�                                    Bx���Z  �          A(��8��A ��?��AG�Bۏ\�8��A ��?�ABffBۀ                                     Bx���   �          A���L��@��R?�=qA@��B�Ǯ�L��@�
=?��A<(�B߽q                                    Bx���  �          A�����@��þB�\��p�B�p�����@��þaG����HB�p�                                    Bx��L  �          AQ����\@��Q���p�B�ff���\@��Y����(�B�k�                                    Bx���  R          AQ��u@�����a�B�.�u@�Q�����c�
B�=q                                    Bx���  V          A�����@�  �L�Ϳ��B�p�����@�  �aG�����B�p�                                    Bx��.>  �          A�
�Mp�@�\�%���
B���Mp�@�=q�'
=���RB��                                    Bx��<�  T          A\)����@�\)���hQ�B�������@�
=�(��ip�B���                                    Bx��K�  �          A\)���H@����G
=����B�(����H@У��G����B�.                                    Bx��Z0  �          A(����@�G��Mp����\CY����@�G��N{����C\)                                    Bx��h�  �          A���33@�Q���  ���C)��33@�Q���  �C�                                    Bx��w|  �          A����G�@���  �H�C�
��G�@���  �H
=C�                                    Bx�ʆ"  �          A\)��@Tz��ʏ\�C\)C&f��@U��ʏ\�C�C\                                    Bx�ʔ�  �          A  �,��@�(�?��A��B����,��@�(�?��A�HB���                                    Bx�ʣn  �          A��Dz�@�p�?�A733Bߏ\�Dz�@�p�?ٙ�A9Bߙ�                                    Bx�ʲ  �          A	��7
=A z�?\)@q�B���7
=A z�?
=@~{B�#�                                    Bx����  �          A	���,(�A?\(�@���Bب��,(�A?fff@��Bخ                                    Bx���`  T          A	���:=q@��
@��A�33B��)�:=q@�@(�A�33B��f                                    Bx���  �          A	����R@�@��
A��B׸R��R@��@��A�Q�B��
                                    Bx���  �          A{�  @��H@�
=A�  B�8R�  @��@�Q�A���B�\)                                    Bx���R  �          Az���  @�  =�Q�?
=B�����  @�  >\)?k�B���                                    Bx��	�  �          A�
�tz�@��H>��?�G�B���tz�@��H>W
=?���B���                                    Bx���  �          A
=���R@��H��\)��G�B������R@��H�#�
�L��B���                                    Bx��'D  �          A��r�\@��\����FffB��r�\@��\����*�HB��                                    Bx��5�  �          A�2�\AQ쿨���	G�B�G��2�\Az῞�R�p�B�=q                                    Bx��D�  �          A���>{A
=�������B����>{A33��G���z�B�                                    Bx��S6  �          A
�H�C33A���A�B�L��C33A�\��RB�G�                                    Bx��a�  �          A\)�AG�A�\��G��@  Bܣ��AG�A�\���.{Bܣ�                                    Bx��p�  �          A���n{A(�>\)?c�
B�z��n{A(�>k�?�p�B�                                     Bx��(  �          A�\��{A  �#�
��\)B잸��{A  =L��>���B잸                                    Bx�ˍ�  �          A
=���A33=��
>�B�8R���A
=>8Q�?��B�8R                                    Bx�˜t  �          A�H���HA��>��@5B��H���HA��?z�@`��B��                                    Bx�˫  �          Ap���ffA	��>��?�ffB����ffA	p�>\@��B�#�                                    Bx�˹�  �          AG����A
=q>�G�@'�B�����A
{?\)@W
=B���                                    Bx���f  �          A��g�Ap�>�33@z�B��=�g�AG�>�@7
=B��\                                    Bx���  �          A���uA��?У�A=qB�ff�uAQ�?�\A+33B�                                    Bx���  �          A����A	p�?�=q@�{B�3���A	�?��H@�Q�B���                                    Bx���X  �          A����RA��?.{@��B� ���RAz�?Q�@��B�\                                    Bx���  �          A��g�A33?�\@G
=B�#��g�A
=?(��@�Q�B�.                                    Bx���  �          A�R�fffA	��:�H���B�\�fffA	p��z��c�
B�                                     Bx�� J  �          Ap���{AG�?�Q�A�Bힸ��{A ��?˅A��B�Ǯ                                    Bx��.�  �          A��(�A (�?��A��B����(�@�\)?��A�
B�B�                                    Bx��=�  �          A(����\@�G�?��A�B�\���\@�Q�?��HA��B�q                                    Bx��L<  �          AG�����@�
=?�(�A��B�(�����@�?�\)A�
B�\)                                    Bx��Z�  �          AG����
@���?ٙ�A(  B�����
@��?�{A7�
B�\)                                    Bx��i�  �          A�H���
@���?��A�HB��=���
@���?�  A�HB��q                                    Bx��x.  �          A\)��  @�(�?p��@�\)B��{��  @�?�{@�
=B��q                                    Bx�̆�  �          A����=q@��R?�@�ffB�����=q@�{?�=qA  B��)                                    Bx�̕z  �          A(����
@�z�?�  @�\)B�����
@��
?�
=@陚B��)                                    Bx�̤   �          A���@�{?^�R@��
B����@��?�ff@�
=B�{                                    Bx�̲�  �          A  ���
@��?��\@��HC�����
@�G�?�Q�@�(�C��                                    Bx���l  �          A=q��G�@�z�?333@��C0���G�@�(�?aG�@��CE                                    Bx���  �          AQ�����@�\?�@P��B�G�����@��?5@�(�B�ff                                    Bx��޸  �          A�\��33@Ϯ��{��\C�\��33@�  �.{��ffC��                                    Bx���^  �          A�R��{@��
�B�\��33C	n��{@�(������C	k�                                    Bx���  �          A\)���@�{>���?��C	���@�>�@3�
C	\                                    Bx��
�  �          A���=q@��H?W
=@���C	���=q@��?�G�@�C	5�                                    Bx��P  �          A  ���H@�{?!G�@x��C	޸���H@�p�?L��@�{C	�3                                    Bx��'�  �          A{��z�@ȣ�?Q�@�Q�C	�q��z�@Ǯ?}p�@�=qC	ٚ                                    Bx��6�  �          A���{@Å?0��@�C����{@\?\(�@�
=C                                    Bx��EB  �          A����Q�@�?��@g
=C	����Q�@��?G�@�
=C	�
                                    Bx��S�  �          A���=q@��?\(�@��C����=q@�(�?��@˅C��                                    Bx��b�  �          Az���Q�@�(�?:�H@�p�CJ=��Q�@�?s33@�ffCc�                                    Bx��q4  �          Aff��ff@��?B�\@�Q�C =q��ff@���?}p�@��C T{                                    Bx���  �          A=q���R@��?G�@�z�C aH���R@�  ?�G�@���C z�                                    Bx�͎�  �          A=q����@�33?Y��@��\B�8R����@�=q?���@�G�B�k�                                    Bx�͝&  �          AG����@�{?��\@��HB�B����@���?�G�@�B��                                    Bx�ͫ�  T          A����\@�{?�z�A�HB��f���\@���?�33A�\B�=q                                    Bx�ͺr  �          AG���
=@�{@�\AEp�C���
=@��
@�A\Q�C�3                                    Bx���  �          A=q���@�\)@�RAm�B�8R���@���@.{A��B��
                                    Bx��׾  �          Aff���@��R?�=q@��B�����@��?���A��B�=q                                    Bx���d  �          A{���A   ?�=q@�
=B�{���@��R?���AB�W
                                    Bx���
  �          A���@���@ ��AB{B�L���@��R@G�A\Q�B���                                    Bx���  �          Az�����@�?�=qAz�B�(�����@��
?�A2�\B��{                                    Bx��V  �          A(���(�A ��?��HA>ffB��
��(�@�
=@\)AZffB�L�                                    Bx�� �  �          A  ����A ��@�AK�B�������@�
=@�Ah  B�G�                                    Bx��/�  �          A����z�@�ff��
=�)p�B�
=��z�A (�����  B�3                                    Bx��>H  �          AG��J=q@�z��s33����Bߨ��J=qA z��`����G�B�                                      Bx��L�  T          A�x��@�=q�Vff���
B����x��@��C�
����B�G�                                    Bx��[�             A33�|(�@���j�H��G�B����|(�@��
�XQ����B�33                                    Bx��j:  	�          A
=�~�R@�{�p  ���B�q�~�R@�=q�]p���{B��                                    Bx��x�  �          A���<(�@�����Q���G�Bި��<(�@�ff��
=���HB���                                    Bx�·�  �          A=q�X��@���fff��Q�B�  �X��A��R�\���B�Q�                                    Bx�Ζ,  V          A�R�y��AG��7���{B癚�y��A�H�#33�{�B�                                      Bx�Τ�  "          Aff���HA ���'����B��f���HA{�33�b�HB�W
                                    Bx�γx  T          A�
�o\)A�Ϳ�33� Q�B�B��o\)A	�����p�B���                                    Bx���  
�          A{�w�A
�\�У��Q�B���w�A����
���B��
                                    Bx����  �          A���z�A  ��G��)�B��f��z�A	���z��\)B�\                                    Bx���j  �          A����33AzῸQ��
=qB�\)��33A	p������Q�B�{                                    Bx���  "          AG���=qA
{�Q���p�B�z���=qA
�\���0��B�W
                                    Bx����  �          A��xQ�A
�H��ff�ȣ�B�33�xQ�A��.{��=qB�                                    Bx��\  �          A��h��AG��������HB��h��A녿:�H���HB��{                                    Bx��  �          A���|(�A33�W
=��Q�B��|(�A����1G�B䞸                                    Bx��(�  "          AG����A�׾\��B�\���A�ͼ#�
��\)B�                                    Bx��7N  �          A  �o\)A��Ǯ�ffB�aH�o\)A��#�
�L��B�W
                                    Bx��E�  "          A(��c33A�׽�G��&ffB�  �c33A��>�\)?�Q�B�                                      Bx��T�  
�          A��aG�A  ����Tz�B��H�aG�A(��#�
�uB���                                    Bx��c@  "          A33�j�HA
=���I��B��
�j�HA33���G�B�Ǯ                                    Bx��q�  T          A��\(�A�Ϳ��]p�Bޮ�\(�A�þ.{��ffBޙ�                                    Bx�π�  �          A{��A ��?�{A�HB����@��?�(�A&ffB�\)                                    Bx�Ϗ2  �          A�H�k�A��?�
=@�G�B�\)�k�A  ?�=qA\)B��                                    Bx�ϝ�  T          A  ���RA�?��
Az�B�����RAff?�z�A4��B�                                    Bx�Ϭ~  �          A���Q�@��
@	��ALz�B��=��Q�@�Q�@ ��Ap  B�G�                                    Bx�ϻ$             A�R���@ڏ\@8Q�A�G�C���@�{@L��A�33C��                                    Bx����  V          A=q���@���@'
=A|  C�3���@�z�@;�A�C��                                    Bx���p  
�          Ap���p�@��@!G�At��B�����p�@�z�@7�A��C z�                                    Bx���  �          AG����
@�Q�@1�A��C����
@��
@G
=A�  C��                                    Bx����  $          A���Ӆ@�G�@W�A��HC	�)�Ӆ@��
@j=qA�C
k�                                    Bx��b  �          A����G�@�=q@\)A�(�C  ��G�@��
@���AӅC�R                                    Bx��  T          A����(�@�z�@8��A�p�C}q��(�@׮@N{A�z�C�                                    Bx��!�  T          AG���z�@�z�@�Ak�C���z�@�Q�@1�A�33C�                                    Bx��0T  T          A(����@�p�@�A^�HB�33���@陚@(��A�B��                                    Bx��>�  �          AG����\AG�?�(�A?\)B�{���\@�\)@�Ag\)B�q                                    Bx��M�  T          Aff�HQ�A�R�
=�k
=BܸR�HQ�Az��
=�?�B�=q                                    Bx��\F  T          A����AQ��8����G�Bѽq��Aff�(��m��B�Q�                                    Bx��j�  T          A(��c�
A	p����/�
B�
=�c�
A
�R��\)�z�B��                                    Bx��y�  �          A��X��A
=��G���Bޞ��X��A(�������\)B�W
                                    Bx�Ј8  �          A��ffA(��˅�Q�B�\��ffA	G���33�ۅB�.                                    Bx�Ж�  �          A�\�}p�AQ�0����33B䙚�}p�A�׾k����B�z�                                    Bx�Х�  
�          A  �|(�A=q��G���RB�Ǯ�|(�A{>�Q�@�B���                                    Bx�д*  
�          A����Q�A
�H>�  ?��RB��)��Q�A
�\?5@�p�B�                                      Bx����  R          Az���33A�>��
?�\)B���33A33?E�@�  B�3                                    Bx���v  V          AG�����Azἣ�
��Q�B�����AQ�>�G�@#�
B���                                    Bx���  �          AG���(�A���  ��33B��f��(�A(�����L��B��                                    Bx����             A�R��A�Ϳ�  ����B�����A	p��
=q�G
=B�q                                    Bx���h  
�          A�H��  A�H��
�?�
B�����  Az�������B�=q                                    Bx��  �          A!G��n�RA\)�%�n�RB�=q�n�RAp���@��B�                                    Bx���  �          A�R�w
=Az��mp����HB��w
=A��P  ��(�B�                                    Bx��)Z  T          A\)�fffA���}p�����B����fffA���`������B��f                                    Bx��8   T          A��N�R@������ٮB���N�R@�(��o\)��ffB��H                                    Bx��F�  
�          A
=q��@�G�>u?�{B�{��@�Q�?!G�@��RB�B�                                    Bx��UL  �          @�\)����@|���n{�$B�=q����@���`  ��
B�Q�                                    Bx��c�  "          @��@G��s33��p�u�C���@G������
=�=C���                                    Bx��r�  �          @�33?�(���z���ffC�>�?�(�=�G����
�q@mp�                                    Bx�с>  �          @��
?�  ������
C���?�  >#�
���
.@��R                                    Bx�я�  "          @�Q�?�?�R�����=A�33?�?��
��\)�)A�p�                                    Bx�ў�  �          @�zᾣ�
@E��p�B��׾��
@^{��\)�w
=B�ff                                    Bx�ѭ0  T          @���l(�@�(���
=�g\)B��
�l(�@�
=��\)�<  B���                                    Bx�ѻ�  �          @���@�ff�Q��\C����@�����b�\CxR                                    Bx���|  �          @�������@���Tz���p�C�\����@��Ϳ����=qCT{                                    Bx���"  �          A ���˅@��H�xQ���G�C.�˅@�z�333���HC�f                                    Bx����  
�          A ����p�@��
�*�H���C���p�@����=q���CǮ                                    Bx���n  
�          @����l(�@�
=�u���33B���l(�@�{�`  ����B��                                    Bx��  
(          @�
=��@�ff����P  B����@��ÿ��R�G�B�u�                                    Bx���  �          @��R�u@��>�\)@
=B=�u@���?:�H@��RB{                                    Bx��"`  �          @�G��J=q@��R>�z�@Q�B��׿J=q@�?=p�@�  B��                                    Bx��1  �          @�����\@��R>W
=?ǮB�Ǯ���\@�{?+�@�p�B��
                                    Bx��?�  T          A�H�˅@��R�k���\)Bʅ�˅@��R>�  ?�p�Bʅ                                    Bx��NR  �          @�(��?\)@�33��
�~=qB���?\)@ָR��z��L(�B��                                    Bx��\�  �          @�=q��\)?�=q��{�G�RC%�3��\)?�����
�C�HC!�
                                    Bx��k�  R          A��C33@��
�
�H�z�HB�p��C33@��޸R�H  B�                                    Bx��zD  
�          A����H@�z�����yBиR���H@�  �޸R�D  B�L�                                    Bx�҈�  T          A���  @���Z=q���
B��ᾀ  @�Q��=p���{B��H                                    Bx�җ�  
(          A�H��z�@�=q�o\)��z�B���z�@����U����B�ff                                    Bx�Ҧ6  
�          A��?
=A z��Q��=B���?
=A녿�����B��3                                    Bx�Ҵ�  
�          A�?\)@�ff���XQ�B�  ?\)A �Ϳ�
=� z�B��                                    Bx��Â  T          A�
>k�@�Q���G���B�L�>k�@��g
=�ӮB�k�                                    Bx���(  
�          Aff<#�
@����G����HB��
<#�
@��R�(�����RB��)                                    Bx����  
�          A�>aG�@�(���Q��G�B�k�>aG�@�����=q��=qB��\                                    Bx���t  "          A��@XQ�@�����"�
B_��@XQ�@�������  Be��                                    Bx���  J          Ap�AQ�>Ǯ��\)��z�@*�HAQ�?&ff��ff�ָR@�
=                                    Bx���  T          A\)A�?�p��������
A<(�A�?�p��|(��ŮAV=q                                    Bx��f  �          A(�A  ?��R�[���  AY�A  @���S33����Ao
=                                    Bx��*  4          A�
A
�R@L���Dz�����A�Q�A
�R@X���7
=���RA��H                                    Bx��8�  �          A�
A�@7��Y����{A�\)A�@E��Mp���z�A�p�                                    Bx��GX  T          A{@���@K����
���A��@���@[��z�H���
A��\                                    Bx��U�  �          Az�@��H@R�\��ff���A�@��H@c33�~�R��33Aȣ�                                    Bx��d�  �          A�@Ӆ@����e��ffB33@Ӆ@���R�\��33B{                                    Bx��sJ  �          A	p�@�
=@�=q��(��=qBA�\@�
=@�33������z�BGp�                                    Bx�Ӂ�  �          A\)?
=@�p������3  B��?
=@�����Q��$�B�.                                    Bx�Ӑ�  �          A��?p��@����p��9ffB���?p��@�{�����+  B��q                                    Bx�ӟ<  �          A
�\?���@�p���Q��9p�B��?���@����(��+�B��                                    Bx�ӭ�  �          A��?u@�{���
�M�B�{?u@�33�����?�B��{                                    Bx�Ӽ�  �          A�R��(�@�  ��p��^�B�(���(�@����
�P�B�(�                                    Bx���.  T          A{��G�@l����ff�}��B��׾�G�@�p���ff�o  B��                                    Bx����  �          AQ��
=@i�����)B�
=��
=@�z����s{B��                                    Bx���z  �          A�H>#�
@��H��(��[Q�B��>#�
@����љ��L��B�p�                                    Bx���   �          AQ�&ff@�����=q�UffB¨��&ff@��R��\)�FB��                                    Bx���  T          A
�H��ff@�����p���33B��ÿ�ff@陚�}p���  B�W
                                    Bx��l  �          AQ��A�N�R��33B��\��A���,������B�ff                                    Bx��#  �          A
=q�J=q@�\������=qB�녿J=q@��\�aG����RB��\                                    Bx��1�  �          A
ff��\)@���q���\)B�8R��\)@�z��Q���{Bǣ�                                    Bx��@^  �          A������@����{�|(�BÙ�����@�����H�A�B�\)                                    Bx��O  �          Az���@�\)���R�B�Ǯ���@����  ���
B�z�                                    Bx��]�  �          A���\)@����j=q���
B˅��\)A   �H������B��H                                    Bx��lP  T          AG��@�\��p��(�B���@��
��ff�噚Bӳ3                                    Bx��z�  �          Aff�.�R@�33��(���{Bܽq�.�R@�33�h����  B�z�                                    Bx�ԉ�  �          A33�G�@�p������
=Bԏ\�G�@�����G�B�33                                    Bx�ԘB  �          A��
�H@ָR��{���B��f�
�H@�=q����\)B�33                                    Bx�Ԧ�  �          A�� ��@�\)���H��ffB�B�� ��@�  �w
=���B���                                    Bx�Ե�  �          A���P  @�\)���R�	p�B�L��P  @ٙ��������
B�.                                    Bx���4  T          A����\)@�p���G��G��B�aH��\)@��
��z��8�\B�                                      Bx����  
�          AQ��@����  �>{B��\��@�33����/
=B��                                    Bx���  �          A(�>�@��H���H�M  B�z�>�@ҏ\���=�B���                                    Bx���&  �          A{?W
=@��
��\�ap�B�B�?W
=@�z���G��R�B���                                    Bx����  �          Aff�B�\@�z��w��ĸRB�(��B�\A{�U��z�B�
=                                    Bx��r  �          A�\�fffA ���S33��  B���fffA  �0����Q�B��                                    Bx��  �          A���s�
@��\�Tz�����B���s�
A z��2�\���B���                                    Bx��*�  �          Aff��{@�p��2�\��z�B�R��{Ap�����_\)B�q                                    Bx��9d  �          A����G�@��R�7
=���B�=q��G�@�(���j=qB�(�                                    Bx��H
  �          Az�����@��N{��ffB�=q����A��,(���p�B��                                    Bx��V�  �          A  �Z=qAff�@  ��ffB�=q�Z=qA	G��(��lz�B�ff                                    Bx��eV  �          A33�33A{>�=q?�(�B�  �33A��?\(�@�p�B��                                    Bx��s�  �          A�\�ٙ�A�\�^�R��\)Bɨ��ٙ�A33��\)��  Bɏ\                                    Bx�Ղ�  �          A\)���A����
�b�RB����A����H�&ffB��
                                    Bx�ՑH  T          Azῥ�Az��'
=����BĸR���A
=�G��I�B�p�                                    Bx�՟�  �          A{���\AQ�#�
�z�HB��f���\A(�>�G�@2�\B��                                    Bx�ծ�  �          A��AG�@����G
=��\)B�k��AG�@�
=�%����B݀                                     Bx�ս:  �          Az���@�33�6ff��ffB�\��A Q����z{B�k�                                    Bx����  �          A\)@9��@�z�����L�RBmQ�@9��@��H��
=�?z�Bv�                                    Bx��چ  �          Ap�@/\)@~�R���j�B^
=@/\)@�  �����]Bj                                    Bx���,  �          A��?�G�@�Q����H�<��B���?�G�@�ff����-��B��
                                    Bx����  �          A(�>�
=@��H���(
=B�
=>�
=@����R��HB�k�                                    Bx��x  �          A�>��R@��
��z��/B���>��R@�����{� ��B���                                    Bx��  �          A  >\)@�(����0p�B��=>\)@�G���\)�!=qB��                                    Bx��#�  �          A�>�=q@ȣ�����:�RB��>�=q@�ff��\)�+�\B�k�                                    Bx��2j  �          A{�+�@�\)��p��B����+�@�33�����B�{                                    Bx��A  T          A녿!G�@�G����\�{B��!G�@�����\�
�B�u�                                    Bx��O�  T          A\)�(�@�\)�����B�Q�(�@��H�����\B��f                                    Bx��^\  S          A���
=@��������B�����
=@�(������
�\B�Q�                                    Bx��m  	�          A  �aG�@���������HB���aG�@����G���
B�k�                                    Bx��{�  �          A33�W
=@�R��{�  B�\)�W
=@�=q��ff�	�
B�33                                    Bx�֊N  
�          A��?!G�@�{����ffB�  ?!G�@�����p��Q�B�p�                                    Bx�֘�  
�          A�>���@�\��33��
B��>���@�(������p�B��                                    Bx�֧�  �          AG����@����������B�G����@��H��Q���ffB�
=                                    Bx�ֶ@  
�          A
ff��@������\����B��)��@�G��dz�����B���                                    Bx����  
�          A�
��Q�@�ff�2�\���B�Ǯ��Q�@��
���|��B�G�                                    Bx��ӌ  �          A	���(�@�p��5����BŊ=��(�Ap��33�{�B�.                                    Bx���2  
�          Ap��	��A���Q����HBѸR�	��A���/\)���B�
=                                    Bx����  �          AQ쾸Q�@�
=�����+=qB��3��Q�@ҏ\��33�33B�Q�                                    Bx���~  �          A=q?�@����33�&�\B�#�?�@ʏ\��{��\B���                                    Bx��$  �          A�@!�@�\)�أ��A
=B�z�@!�@����z��3G�B��3                                    Bx���  �          A��@	��@�{��p��G
=B��@	��@�z��љ��9  B��H                                    Bx��+p  
�          Ap�?�(�@����Ӆ�:��B�#�?�(�@�=q�ƸR�,�B�p�                                    Bx��:  "          Ap�?�
=@��ə��/��B�B�?�
=@ڏ\���
�!
=B�u�                                    Bx��H�  
�          Ap�?��
@��������'
=B�8R?��
@�����33�Q�B�aH                                    Bx��Wb  �          Ap�?޸R@��H���H���B�p�?޸R@������B���                                    Bx��f  
�          A��?!G�@�33��
=�=qB�=q?!G�@�p���\)���RB���                                    Bx��t�  T          A�ý�G�@�33��=q��B��)��G�@�ff��33�	��B�                                    Bx�׃T  �          A�
>�\)@��������-
=B�#�>�\)@�����R�(�B�ff                                    Bx�ב�  �          A�þW
=@�G�����%��B����W
=@��������RB�p�                                    Bx�נ�  
�          A���˅@�
=���\� �\BΨ��˅@�\��(���B�W
                                    Bx�ׯF  "          A�\�\(�@�������33B��=�\(�A����33�ծB��                                    Bx�׽�  
�          A33���
@�\����G�B�\)���
@�p���  �z�B�W
                                    Bx��̒  	�          A�\���@�{�h���£�B��)���A�\�G���p�B�Q�                                    Bx���8  �          A�����@���������RB��ÿ��@��H��G��ݮB�B�                                    Bx����  
�          A�ff@�����vffB�B��ff@��ÿ�{�>=qB׸R                                    Bx����  
�          A�R�33@�������RBծ�33@�Q�����Y�B��                                    Bx��*  �          A�H��H@�R���H���HB�#���H@����
��
=B��
                                    Bx���  
Z          AG����@����\��B�
=���@�{���
�ffB��                                    Bx��$v  
(          A(���
=@�p���ff��B�G���
=@����Q��33B��                                    Bx��3  �          A
=�h��@�z���p��.�B�33�h��@أ���  � z�B�L�                                    Bx��A�  "          A(����H@�������B����H@�
=���H��G�B�#�                                    Bx��Ph  �          A������@�����{��HB��f����@�=q�����\)B���                                    Bx��_  "          A���b�\@��H��
=�I�B���b�\@�{��
=���B�{                                    Bx��m�  �          Azῥ�A �׿����{B�33���A���!G����
B�{                                    Bx��|Z  �          A�
�-p�@��
���
��
B�#��-p�@��L�����HB��
                                    Bx�؋   �          A��qG�@��>�33@   B�=�qG�@�(�?L��@�{B�Ǯ                                    Bx�ؙ�  #          A33�?\)@�p���33��HBފ=�?\)@�  �k��ʏ\B�(�                                    Bx�بL  !          A{�z�H@�G�?(��@��B�\�z�H@�?�{@��RB��                                    Bx�ض�  	�          A������@˅@<(�A�p�B�#�����@��@U�A�
=B���                                    Bx��Ř  "          A�
�g
=@�ff?��A�B�B��g
=@�33?�  AFffB��H                                    Bx���>  U          A\)�@��@���?:�H@�G�B�L��@��@�
=?�(�A��Bޙ�                                    Bx����  �          A	G��h��@��Ϳ��\����B���h��@�{����l(�B��                                    Bx���  �          Ap��j�H@����{�陚B�
=�j�H@����R���HB�q                                    Bx�� 0  �          Ap��g
=A�H?�Q�A{B�\�g
=AG�?�Q�AE��B��                                    Bx���  
Z          A���u�@��?���A"ffB���u�@�  @�\AT��B�{                                    Bx��|  S          A���G�A   ?�Q�A+
=B�{��G�@�(�@�A\��B�B�                                    Bx��,"  �          A���fffA�׾�\)��B��{�fffA��>��?ǮB��{                                    Bx��:�  �          Ap���A��=L��>���B�����A��?��@UB��                                    Bx��In  �          A������@�\)���\��  B��
����A �׿G�����B�p�                                    Bx��X  �          A�R����@���>��@p�B�������@�Q�?aG�@���B��                                    Bx��f�  �          A����  AG��xQ���
=B�q��  A{��G��$z�B�                                    Bx��u`  T          Ap���ffA�Ϳ��\���B�  ��ffAp����H�8Q�B�                                    Bx�ل  T          A���y��Aff���B�\B�\)�y��A{?   @<��B�k�                                    Bx�ْ�  
�          A��{@��R?�G�@���B���{@��
?޸RA+�B�                                    Bx�١R  
�          A��dz�A
=@��AX��B���dz�A��@1�A��B�W
                                    Bx�ٯ�  �          A\)��33A�?��\@�z�B����33A
=q?��
A=qB�\                                    Bx�پ�  �          A���=qAff@#33Am��B�����=q@�\)@AG�A��B��H                                    Bx���D  
�          A��j=qA�@i��A���B��j=qA  @�z�A�\)B��                                    Bx����  
�          A��dz�A
=@�z�A�p�B�{�dz�@��@��A�  B�                                     Bx���  �          Aff�R�\@�=q@��HA�B�3�R�\@�Q�@�G�B33B�ff                                    Bx���6  "          A{���\A�?
=q@N�RB�ff���\A Q�?�G�@���B��3                                    Bx���  	�          Aff��33@�ff?Tz�@�{B��
��33@�z�?��@�{B�B�                                    Bx���  
�          A���  A33?���A��B�R��  A��?�A6{B�W
                                    Bx��%(  
�          Ap���z�@�ff@��Ai��B���z�@���@6ffA��B�{                                    Bx��3�  
�          A{����@��@�p�A�=qB��H����@��@���A�Q�B��)                                    Bx��Bt  T          A����@�\@AG�A��B��f���@���@Z�HA��\B�Q�                                    Bx��Q  T          Ap���p�@��
@2�\A�z�C+���p�@�ff@EA���C�                                    