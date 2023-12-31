CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230420000000_e20230420235959_p20230422014049_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-22T01:40:49.800Z   date_calibration_data_updated         2023-04-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-04-20T00:00:00.000Z   time_coverage_end         2023-04-20T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxy�_@  	�          A=q@��@�z�@'�A�BGQ�@��@�{@�B
=B��                                    Bxy�m�  �          A\)@�=q@�@1G�A�  BG@�=q@���@��\B
=B�\                                    Bxy�|�  �          A
=@��H@�(�@/\)A�33BF�@��H@�(�@���B33B�\                                    Bxy��2  �          A�H@��R@�@8Q�A�Q�BJ{@��R@�33@�p�BB�\                                    Bxy���  �          A�@���@ۅ@1�A��RBE(�@���@�33@��B=qB�\                                    Bxy��~  �          A��@��\@�
=@9��A�BH=q@��\@�(�@��RB�RB�R                                    Bxy��$  T          A��@��@��@FffA��BG@��@�\)@�33B  B��                                    Bxy���  T          A�@�33@�z�@.{A��
B9(�@�33@�{@�z�B33B�
                                    Bxy��p  "          A��@�  @��@>{A�p�B0�R@�  @���@�ffB��A�z�                                    Bxy��  
�          A�@�33@��
@8��A��BAp�@�33@���@���B{B�
                                    Bxy��  �          A�@�{@�  @7�A��HBF�R@�{@�@�{B��B�                                    Bxy� b  	�          A��@��H@��@1G�A�
=BI  @��H@�  @��
B�RB�                                    Bxy�  
�          A\)@��@�\@2�\A�33BNff@��@���@��B�RB"                                      Bxy��  
�          A\)@�z�@�  @<(�A��RBL�@�z�@���@�  Bp�B
=                                    Bxy�,T  "          A  @�\)@�@C33A��BI�R@�\)@�G�@��B��B��                                    Bxy�:�  �          A(�@��@أ�@G�A�\)BC�@��@��@���B��B��                                    Bxy�I�  �          A(�@�ff@�\)@H��A�(�B9�@�ff@��@��B�B(�                                    Bxy�XF  "          A  @�
=@���@?\)A���B9��@�
=@�
=@��B�HB�                                    Bxy�f�  
�          A\)@ƸR@�G�@7�A�G�B9�@ƸR@�G�@�ffB\)B
��                                    Bxy�u�  
Z          A�@�ff@���@,��A�Q�B<  @�ff@�
=@��B�B=q                                    Bxy��8  "          A(�@�Q�@�@$z�AzffB;p�@�Q�@�=q@�Q�B��Bz�                                    Bxy���  T          A��@�
=@أ�@%�Az�HB=��@�
=@�z�@�=qB��B                                      Bxy���  �          A��@��@�=q@$z�Ay��B?�@��@�{@��\B
=B=q                                    Bxy��*  �          A�@�z�@��@��Amp�BA��@�z�@��\@���B��B�H                                    Bxy���  
�          A��@�
=@��@{AU�BH�R@�
=@�p�@�{B
(�B#ff                                    Bxy��v  X          A33@�G�@��H@,��A�G�BFQ�@�G�@��
@�=qBz�Bz�                                    Bxy��            A�@���@�
=@ffA`(�BH�@���@���@��\B=qB!�                                    Bxy���  
�          A33@Å@���@ffA`��BF  @Å@�33@���B�BQ�                                    Bxy��h  
�          A�@�G�@�Q�@>�RA��RBE  @�G�@�p�@���B
=B��                                    Bxy�  �          A
=@���@�z�@�RAn{BGQ�@���@���@���B  B=q                                    Bxy��  "          A�\@�
=@�p�@�RAn{BH�@�
=@���@��B�\B!
=                                    Bxy�%Z  T          A��@��@�{@W�A���BL�@��@�p�@�=qB&{B��                                    Bxy�4   �          AQ�@��@�@��Ai�BM�\@��@��@��\B  B&��                                    Bxy�B�  
�          Aff@�(�@��@UA�{BM@�(�@���@�33B%  B                                    Bxy�QL  "          AG�@���@�Q�@���Aԏ\BP33@���@tz�@���B<��B�
                                    Bxy�_�  T          A(�@�G�@��
@=p�A�\)BP\)@�G�@���@���B33B#(�                                    Bxy�n�  T          AQ�@�z�@�{@N{A�ffBK��@�z�@�  @�{B"ffBp�                                    Bxy�}>  
�          A��@�p�@�33@^{A�z�BIz�@�p�@��@˅B'  B��                                    Bxy���  "          A  @�(�@�
=@h��A��\BH�@�(�@��@�ffB+
=Bz�                                    Bxy���  �          A�
@�z�@�\)@fffA�ffBH=q@�z�@�z�@��B*
=BG�                                    Bxy��0  �          A�
@��@ָR@g
=A�
=BGQ�@��@��
@��B)�B{                                    Bxy���  �          Az�@��R@�ff@h��A��BFQ�@��R@�33@�B)�RB�H                                    Bxy��|  T          Az�@�33@�  @n�RA�Q�BI\)@�33@�33@���B-  B                                      Bxy��"  &          Az�@��\@���@mp�A�\)BJ(�@��\@�z�@���B,�
B=q                                    Bxy���  �          A��@��H@ٙ�@j�HA��BJG�@��H@�@�  B+�
B�                                    Bxy��n  
�          A��@��R@�ff@l��A�=qBF=q@��R@��\@�
=B*�RB=q                                    Bxy�  "          A@�Q�@��@c33A�\)BG�@�Q�@�Q�@���B'  B��                                    Bxy��  
�          A@�z�@�
=@aG�A��
BC  @�z�@�{@�=qB$��B��                                    Bxy�`  �          A{@��R@�p�@b�\A���B@�R@��R@�(�@��B$�B
�                                    Bxy�-  T          A�@��@�p�@a�A�Q�BJ�@��@��
@�B'�
B��                                    Bxy�;�  �          A�@��@���@e�A���BE��@��@�
=@���B&�
B
=                                    Bxy�JR  �          Aff@��@�
=@X��A��RBJ
=@��@�\)@ʏ\B#��B��                                    Bxy�X�  
^          A33@�G�@߮@Z=qA��\BIp�@�G�@��@�33B#��B33                                    Bxy�g�  P          A
=@�  @�  @\(�A�(�BJp�@�  @��@�(�B$��B��                                    Bxy�vD  �          A33@���@߮@\(�A�  BIz�@���@�\)@��
B$(�B                                      Bxy�  �          A��@��
@�\@Y��A�=qBIp�@��
@��\@�(�B"\)B                                      Bxy�  T          A��@�33@�z�@\��A�BJ�R@�33@��@�ffB#z�B�                                    Bxy¢6  �          A�@�33@��@^�RA��BK  @�33@��@�\)B$(�B(�                                    Bxy°�  T          A�\@���@�p�@\��A���BJ=q@���@���@θRB"B�                                    Bxy¿�  
�          Aff@�z�@�33@fffA��\BIp�@�z�@���@��B%�B                                      Bxy��(  "          A@Å@ٙ�@l(�A��
B@(�@Å@��R@ϮB$�RB
(�                                    Bxy���  �          Ap�@�G�@�z�@dz�A�z�BB��@�G�@��H@�B#Q�B�
                                    Bxy��t  
�          Ap�@���@���@fffA�BC��@���@��H@θRB$�BG�                                    Bxy��  "          A{@�\)@�z�@XQ�A�  B?�\@�\)@�{@�Q�B�B                                      Bxy��  
�          A�H@�=q@�@O\)A�z�B>z�@�=q@���@���B=qB�
                                    Bxy�f  	�          A�@�Q�@�Q�@p��A�
=BEp�@�Q�@��
@�z�B'{B{                                    Bxy�&  
�          A Q�@�z�@�=q@�  A��BH@�z�@��@�(�B-=qB�
                                    Bxy�4�  "          A�@��@�\@W
=A���BC�H@��@�(�@ʏ\Bz�B�                                    Bxy�CX  �          A ��@�z�@�\)@n{A��BKG�@�z�@��H@ָRB'�B                                    Bxy�Q�  T          A Q�@��H@��H@L��A��RB833@��H@�  @��B��B	G�                                    Bxy�`�  
�          A ��@ҏ\@ۅ@N�RA�  B8�@ҏ\@�Q�@�33B�B	�\                                    Bxy�oJ  
�          A ��@�=q@ۅ@Tz�A�Q�B9
=@�=q@�
=@�B��B�H                                    Bxy�}�  T          A!p�@У�@�
=@QG�A�p�B;�@У�@��H@�{B=qB�\                                    BxyÌ�  "          A z�@�p�@�\@`  A�\)BC�
@�p�@��\@�BB�                                    BxyÛ<  T          A�@�  @�R@�33A�z�BR��@�  @�p�@��B3p�B��                                    Bxyé�  
�          A\)@���@陚@z=qA�
=BS��@���@��\@���B/p�B�
                                    Bxyø�  
�          A\)@���@�\@a�A�(�BF�\@���@��\@�ffB!�HB                                      Bxy��.  �          A
=@�  @�\@���A�  BP��@�  @�
=@�B6�HB{                                    Bxy���  '          Aff@��
@�@�33AݮBY{@��
@�33@��BB  B�
                                    Bxy��z            A�H@�@��H@�(�A�z�BWz�@�@�=q@�BA�RB��                                    Bxy��   "          A
=@�p�@�z�@��A���BXff@�p�@�(�@�BAG�Bz�                                    Bxy��  
�          A33@��@�ff@�33A�{BZ�\@��@�{@�ffBA��Bp�                                    Bxy�l  �          A (�@�33@�  @�z�A��HB[z�@�33@�
=@�  BBG�B\)                                    Bxy�  �          A"�H@�Q�@�\@�
=A���BY=q@�Q�@�  @�33BA\)B��                                    Bxy�-�  
�          A!�@��@��@�{A�  BU�@��@��
@�  B@�B33                                    Bxy�<^  "          A"{@���@�Q�@��RA݅BX
=@���@�ff@��BA�BG�                                    Bxy�K  
(          A"{@��@�{@��RAݙ�BUG�@��@���@���B@
=B
=                                    Bxy�Y�  T          A z�@��H@��
@�(�A�BTp�@��H@��
@�p�B>�HB�                                    Bxy�hP  
�          A   @��\@�33@��A�p�BTp�@��\@��
@�z�B>��B�H                                    Bxy�v�  
�          A ��@��
@�=q@�
=A��
BS(�@��
@���@�
=B@{B{                                    Bxyą�  	�          A ��@��R@�  @�\)A�Q�BP{@��R@~�R@�ffB?{BQ�                                    BxyĔB  
�          A!�@�  @�@���A�BN\)@�  @x��@�\)B?�B=q                                    BxyĢ�  	�          A!�@�  @��@�AݮBO�\@�  @���@��B=�\B�R                                    Bxyı�  	�          A!@��H@�Q�@���A�33BM�R@��H@�G�@�(�B;�RBff                                    Bxy��4  T          A!�@�
=@޸R@��A�Q�BO=q@�
=@w�@��HBB(�B�                                    Bxy���  T          A!@�  @�ff@�33A���BS��@�  @q�@��BHp�B�                                    Bxy�݀  
�          A#�
@�z�@ᙚ@��A�p�BRQ�@�z�@w�@���BFffB��                                    Bxy��&  T          A$��@�(�@�p�@�33A�  BK�\@�(�@qG�@�\)BC{B(�                                    Bxy���  
�          A$��@�
=@�p�@���A��HBI�H@�
=@s33@�B@��B\)                                    Bxy�	r  
�          A$��@\@�
=@�33A�G�B?�@\@mp�@��B8�\A�G�                                    Bxy�  
�          A$��@��
@�Q�@�=qA���B-��@��
@Tz�@�p�B0=qA��                                    Bxy�&�  Y          A%�@�  @�ff@�G�A�=qB@��@�  @g
=@�=qB<�HA�=q                                    Bxy�5d            A$z�@��
@�G�@��HA�ffB@G�@��
@U�@�Q�BC�HA�\                                    Bxy�D
  
�          A%G�@�
=@�z�@�p�A�  BI(�@�
=@n�R@���BB�RBQ�                                    Bxy�R�  �          A$��@�ff@�z�@��
A�p�BN�@�ff@h��@�{BI�HB
=                                    Bxy�aV  T          A$z�@��@�\)@�
=A癚BK�
@��@y��@��
B?��B
p�                                    Bxy�o�  �          A$��@��\@��H@�  A�33B[��@��\@�p�@�B:\)B$                                    Bxy�~�  �          A%p�@�@�@�z�A�  BO�@�@���@�ffB9�Bp�                                    BxyōH  
(          A#�@�\)@�33@��A�z�BL\)@�\)@�{@�33B8{B�                                    Bxyś�  T          A"{@�ff@�ff@��
A�BJ��@�ff@���@陚B8�
B�R                                    BxyŪ�  
�          A#�@�ff@��@�  A�Q�B`��@�ff@�G�@��HBHffB#�R                                    BxyŹ:  "          A%�@�(�@���@�  B  BoQ�@�(�@�Ap�BX
=B.�                                    Bxy���  �          A$z�@�p�@��@��A��Bh�\@�p�@�G�A��BO��B*(�                                    Bxy�ֆ  �          A"�H@�ff@���@��HA�{B_=q@�ff@��@�33BJ33B {                                    Bxy��,  �          A"�H@��@�R@�ffA��Bc�\@��@��
@��BH33B'�H                                    Bxy���  �          A#33@�{@�@�A�\)Baz�@�{@��
@���BF�
B%��                                    Bxy�x  T          A#
=@�=q@�ff@�  A���Bdp�@�=q@��@��HBIp�B(�                                    Bxy�  T          A#
=@�Q�@�{@��\A�
=Be�@�Q�@�=q@��BK��B(�                                    Bxy��  "          A#
=@��@�\@�Q�A��RBd�@��@�z�A Q�BO��B$�                                    Bxy�.j  �          A#33@��
@�33@�33A�\)Bg�@��
@��
ABR��B&�                                    Bxy�=  �          A"�H@���@�Q�@��B �Be\)@���@�Q�A�BS�\B#{                                    Bxy�K�  T          A#
=@�\)@���@�Q�B�HBb
=@�\)@w�AffBTz�B\)                                    Bxy�Z\  T          A#�@�p�@�\@��B�
B\�H@�p�@u�A��BQ��B��                                    Bxy�i  �          A"�H@�ff@�{@�=qB��BZ33@�ff@j�HABS{Bp�                                    Bxy�w�  T          A"�\@��
@�\@�p�B  B^
=@��
@w�A ��BQ�B��                                    BxyƆN  
�          A"�R@��\@�=q@��B�RB^��@��\@u�Ap�BR�
B�
                                    BxyƔ�  �          A#\)@�z�@��@��A�33B`�H@�z�@�ff@��BK33B"�\                                    Bxyƣ�  
�          A#�
@��\@���@��BffB^{@��\@n{A�BU��B�\                                    BxyƲ@  �          A$Q�@��@׮@���B=qBX�R@��@S33A
=B\��Bz�                                    Bxy���  �          A$  @��@��@�ffB  BX�\@��@g
=A
=BT33B�                                    Bxy�ό  
�          A$  @��@�\)@�p�B
=BQ�\@��@AG�A�RB]{A�p�                                    Bxy��2  
�          A#\)@��@��H@���Bp�BU\)@��@K�AB\33B�R                                    Bxy���  �          A#�
@�33@��
@��RB\)BLQ�@�33@W
=A�BOB �                                    Bxy��~  
�          A$��@�=q@��H@��HA��BKff@�=q@n{@��HBE�B�                                    Bxy�
$  �          A%��@��@߮@��HA���BG
=@��@��H@�ffB8(�B
��                                    Bxy��  "          A%@��
@�G�@�Q�AΣ�BC�@��
@���@�B/z�B�H                                    Bxy�'p  �          A%�@���@�\)@�(�A�{B@�@���@�G�@���B+
=B	z�                                    Bxy�6  �          A&{@�z�@ٙ�@��HA��HB?�R@�z�@{�@��
B533B�                                    Bxy�D�  "          A'�@�Q�@���@��HA�{B8�@�Q�@���@��
B+��A�(�                                    Bxy�Sb  �          A(Q�@��@�ff@��
B=qBP�@��@5�A��Bcp�A�ff                                    Bxy�b  �          A(Q�@n{@���@�B5Q�BoG�@n{@=qA��B�33B�                                    Bxy�p�  "          A(Q�@s33@�@���B2��Bm�@s33@�RA�
B��HB�\                                    Bxy�T  �          A(  @mp�@У�@��HB1p�Bq�@mp�@&ffA�B��HB                                      BxyǍ�  �          A'�@g�@ҏ\@�G�B0�Bt��@g�@+�A33B�
=B                                      Bxyǜ�  
�          A'�
@e@�{@�{B5
=Bsz�@e@\)AQ�B��3B
��                                    BxyǫF  "          A(  @h��@θR@�{B4p�Brff@h��@ ��Az�B�8RB
\)                                    Bxyǹ�  T          A(  @Tz�@�Q�@�  B>G�BxG�@Tz�@��A�B�p�B(�                                    Bxy�Ȓ  T          A'�@U@�@�Q�B?ffBv�\@U@Q�A
=B��\BG�                                    Bxy��8  �          A'
=@Z=q@��H@�Q�B@
=Br��@Z=q@�
AffB��A���                                    Bxy���  �          A&�R@\(�@��@�  B@�Bq�@\(�@�\A�B��)A�G�                                    Bxy��  
�          A'
=@dz�@��@��HBB�Bk��@dz�?�\)A=qB���A�p�                                    Bxy�*  �          A'33@W
=@�G�@�=qBB�Bs�@W
=@   A�HB�  A�=q                                    Bxy��  
�          A&�R@X��@�G�@���BAG�Br@X��@G�A=qB�u�A���                                    Bxy� v            A&{@b�\@�(�@�G�BBz�Bk�@b�\?��A�B���A�=q                                    Bxy�/  �          A&{@U@�
=@�\)BI33Bn�H@U?�z�A�HB���AӅ                                    Bxy�=�  T          A&�H@HQ�@��
A (�BI33Bw�H@HQ�?��AQ�B��=A�(�                                    Bxy�Lh  �          A)@@��HABb��B��R@?s33A$��B�z�A�Q�                                    Bxy�[  T          A*=q@'�@�33A
ffBY�HB��f@'�?��\A#�B�=qAϙ�                                    Bxy�i�  T          A*=q@8Q�@�\)A�
BSB}ff@8Q�?�(�A"=qB���Aأ�                                    Bxy�xZ  
�          A)@>�R@��
A�BN��B|p�@>�R?�A ��B�p�A���                                    Bxyȇ   �          A)��@8Q�@���A=qBQ\)B~=q@8Q�?˅A!�B���A��                                    Bxyȕ�  "          A*�\@0��@��A
{BX\)B\)@0��?���A#\)B��Ạ�                                    BxyȤL  
�          A*�R@5�@��RA{BO{B�G�@5�?�  A"{B���A�z�                                    BxyȲ�  
�          A*�H@5�@�\)A=qBN��B�\)@5�?�\A"ffB���B {                                    Bxy���  �          A*�H@7
=@�
=A=qBN��B��
@7
=?�G�A"=qB�\)A���                                    Bxy��>  T          A*�H@AG�@�ffA��BM�B|ff@AG�?�\A!��B�ǮA��                                    Bxy���  	.          A*�R@E�@�  AQ�BK=qB{Q�@E�?�{A ��B���A��\                                    Bxy��  �          A*�H@?\)@�  A��BL\)B~
=@?\)?���A!G�B��=A��                                    Bxy��0  �          A*�R@C�
@�=qA�BIB}�@C�
?���A ��B�G�BG�                                    Bxy�
�  �          A*�\@G
=@\A
=BHB{p�@G
=?�p�A   B��B�                                    Bxy�|  T          A*ff@N�R@��HA�BF��Bxff@N�R@�A
=B�
=B                                     Bxy�("  
�          A*�\@O\)@��A�\BG��Bwz�@O\)?��RA\)B�=qA�=q                                    Bxy�6�  �          A)�@L��@�\)A�RBIG�Bw�@L��?�A
=B��fA���                                    Bxy�En  T          A*{@J=q@�z�A(�BL�Bw
=@J=q?�ffA�B��A�p�                                    Bxy�T  "          A)�@L(�@��\A��BM
=Bu\)@L(�?�  A�B���A��                                    Bxy�b�  T          A*{@QG�@���A��BMffBr(�@QG�?ٙ�A�B�z�A�{                                    Bxy�q`  �          A*=q@K�@��
Az�BL��Bv\)@K�?��A�
B��fA�                                    Bxyɀ  T          A*=q@K�@��HA��BMG�Bu@K�?�\A   B�A�Q�                                    BxyɎ�  
�          A*ff@G�@�{AQ�BK��By�@G�?��A (�B�A�{                                    BxyɝR  
�          A*=q@H��@��RA�
BK  Bx@H��?�z�A�
B��{A�Q�                                    Bxyɫ�  
�          A*ff@J=q@�G�A�RBH�\By�\@J=q@�A\)B��B��                                    Bxyɺ�  T          A*=q@HQ�@�33A{BGz�B{=q@HQ�@ffA33B�z�B=q                                    Bxy��D  "          A)�@L(�@�z�A ��BEG�Bz33@L(�@�A=qB�W
B	33                                    Bxy���  �          A*�\@L(�@�33A{BG  By�R@L(�@
=A
=B��B=q                                    Bxy��  �          A*�\@J=q@���A\)BIz�ByG�@J=q@ ��A�B��)B
=                                    Bxy��6  "          A*�H@XQ�@\A��BEp�Bs@XQ�@�AffB�(�B \)                                    Bxy��  "          A*=q@P  @���AffBG�HBvz�@P  @33A�RB���B                                      Bxy��  "          A)p�@G�@��HAG�BF�B{�\@G�@	��A{B�\B
\)                                    Bxy�!(  T          A)G�@J�H@�=qA ��BF��By�
@J�H@	��AB��{B�\                                    Bxy�/�  �          A)p�@L��@��A   BD(�Bz\)@L��@��Ap�B���B(�                                    Bxy�>t  �          A(��@L(�@�ff@�p�BB�\Bz�@L(�@�Az�B���B�                                    Bxy�M  T          A(��@Q�@Ǯ@��HB@33By(�@Q�@=qA�B���BQ�                                    Bxy�[�  
�          A(��@U�@�
=@�=qB?��Bw\)@U�@=qA33B�{Bff                                    Bxy�jf  "          A(��@S33@�@�z�BA�Bw�@S33@A�
B��fB�R                                    Bxy�y  "          A)G�@Vff@���@�{BBQ�Bu��@Vff@�
AQ�B��
B
Q�                                    Bxyʇ�  �          A)G�@W
=@�Q�A z�BEz�Bs(�@W
=@	��A��B��Bff                                    BxyʖX  �          A(��@W�@��@��RBC�HBs�@W�@{A  B�
=B                                    Bxyʤ�  �          A(z�@U@��@��BE�BsG�@U@	��A  B���B(�                                    Bxyʳ�  O          A(��@QG�@���Ap�BH��Bs�@QG�@�\A��B�{A�33                                    Bxy��J  
�          A'�
@P  @�Q�A=qBK�Br33@P  ?�33A��B��HA�R                                    Bxy���  �          A((�@W�@�p�A Q�BF�Bqp�@W�@ffA�
B��A��                                    Bxy�ߖ  
�          A(��@W
=@�z�@�(�BA�Bu\)@W
=@�A33B��B��                                    Bxy��<  "          A)p�@Tz�@Ǯ@�z�B@��Bx  @Tz�@p�A  B��B(�                                    Bxy���  "          A)�@N�R@�\)@��BA��Bzp�@N�R@��A(�B��B��                                    Bxy��  �          A)��@R�\@�  @���B@�
By(�@R�\@�RAQ�B�Q�B(�                                    Bxy�.  �          A)@W
=@љ�@�(�B8�B{ff@W
=@7�A=qB�ǮB!��                                    Bxy�(�  �          A)�@Vff@���@�B9�B{�\@Vff@5A�RB�.B!Q�                                    Bxy�7z  T          A*{@W�@Ϯ@��RB9��BzG�@W�@333A�HB�Q�B                                    Bxy�F   �          A*=q@^{@��
@���B<{Bu�@^{@*=qA33B�p�B�                                    Bxy�T�  �          A)�@X��@��@�B:�\Bx��@X��@0  A�B�=qB33                                    Bxy�cl  �          A(��@W�@�ff@��
B9G�By�@W�@3�
AG�B��qB(�                                    Bxy�r  
�          A(��@Z�H@�\)@�=qB7�RBx��@Z�H@7�A��B��fB                                       Bxyˀ�  �          A)G�@X��@�G�@��HB7p�Bzz�@X��@:�HAp�B���B#
=                                    Bxyˏ^  
�          A)p�@Z=q@�G�@��HB7G�By��@Z=q@;�Ap�B�B"p�                                    Bxy˞  
Z          A)G�@\��@У�@��HB733Bx��@\��@:�HA�B�z�B!                                      Bxyˬ�  �          A)�@Z�H@ҏ\@�G�B5�Bz  @Z�H@?\)A��B�{B$��                                    Bxy˻P  T          A(z�@dz�@ƸR@��RB<ffBpQ�@dz�@%A��B�aHB�H                                    Bxy���  
�          A(��@X��@ҏ\@��B5�HBz��@X��@@��Az�B��B&�                                    Bxy�؜  �          A)G�@J=q@���@�
=B3�
B�  @J=q@N{A��B�k�B6=q                                    Bxy��B  �          A)�@J�H@�p�@�=qB/p�B�Ǯ@J�H@Z�HA�B�B<��                                    Bxy���  �          A)�@N{@��H@�z�B1ffB���@N{@Tz�A  B�{B7                                    Bxy��  �          A)�@J=q@�(�@��
B0�B��R@J=q@XQ�A  B��B;�R                                    Bxy�4  �          A)p�@I��@��@�(�B0��B��@I��@Z=qA(�B��B<��                                    Bxy�!�  
�          A)��@^{@�ff@�
=B+��B}�
@^{@`��A{By
=B5\)                                    Bxy�0�  �          A)��@\��@�
=@�p�B1�HB{=q@\��@N�RA�B}�B,ff                                    Bxy�?&  �          A)��@g
=@�{@��B)�HBz�@g
=@b�\A�Bv(�B1��                                    Bxy�M�  T          A)@dz�@�33@�Q�B,��By��@dz�@Z�HA{Bx��B/=q                                    Bxy�\r  �          A)��@dz�@�@�  B%z�B}Q�@dz�@qG�A�
Br�RB:33                                    Bxy�k  T          A)p�@^{@��@��
B!�B��@^{@~{A�HBpz�BCff                                    Bxy�y�  �          A)�@]p�@�33@�Q�B{B��\@]p�@�33ABn  BGG�                                    Bxÿd  
�          A)�@Mp�@��R@��B�RB��R@Mp�@�Q�A��BjQ�BZ�                                    Bxy̗
  
Z          A)��@B�\@�=q@θRBffB�p�@B�\@��A(�Bi(�Bc\)                                    Bxy̥�  "          A)�@Dz�@�p�@��
Bz�B��=@Dz�@�G�A\)BfG�Be=q                                    Bxy̴V  
�          A*�\@Tz�@���@ʏ\B�
B�p�@Tz�@���A�RBc33B]\)                                    Bxy���  T          A+\)@J�HA z�@�=qB�
B��@J�H@�A\)Bc33Be
=                                    Bxy�Ѣ  �          A+�@HQ�A z�@�z�BG�B�ff@HQ�@���A(�Bd��Be�
                                    Bxy��H  
Z          A+33@9��A��@��HB
�
B��\@9��@���Ap�B^�Bu{                                    Bxy���  �          A+33@0  A	G�@���B��B�u�@0  @��A
ffBW��B�33                                    Bxy���  
Z          A+
=@I��@��R@�(�B�B�Ǯ@I��@��
A\)Bd\)Bd=q                                    Bxy�:  �          A+33@N�R@�@�33B\)B��3@N�R@��A�
Bo
=BT�R                                    Bxy��  
Z          A+�@J�H@�=q@�p�B'�
B�.@J�H@~�RA
=Bv�RBM�                                    Bxy�)�  T          A+
=@G
=@��
@���B;  B���@G
=@G
=A�B�Q�B3�R                                    Bxy�8,  
�          A*=q@P  @�@��B=33B|��@P  @;�A�HB�\)B'�H                                    Bxy�F�  
�          A)�@Z�H@��@��\B=�HBvG�@Z�H@4z�AffB��B{                                    Bxy�Ux  
�          A*=q@^{@�z�@��RBA�Br  @^{@'
=A\)B���B                                    Bxy�d  "          A*ff@c33@\@��BBz�Bn��@c33@#33A\)B�� B��                                    Bxy�r�  
�          A*{@q�@��A�RBH��Ba(�@q�@
=A\)B�  A�33                                    Bxýj  T          A*�\@dz�@���A z�BC�Bmff@dz�@ ��A�B���B=q                                    Bxy͐  �          A*�R@g
=@��A ��BD
=Bk��@g
=@{A�B��{B	��                                    Bxy͞�  
U          A*�\@e@�A��BE�RBk33@e@��A  B�.B{                                    Bxyͭ\  �          A*�R@s33@���A�BFffBb��@s33@��A\)B�
=A��                                    Bxyͼ  
�          A*�H@n{@���A�HBG�Bd��@n{@\)A(�B���A�                                      Bxy�ʨ  �          A*�\@e�@�=qA�HBHp�Bi�@e�@�\AQ�B���B�\                                    Bxy��N  �          A*�\@`  @���A�\BG��BmQ�@`  @Q�Az�B�=qB                                    Bxy���  
�          A*�H@Z�H@���A\)BI{Bo��@Z�H@�AG�B��B
�R                                    Bxy���  
(          A*�\@W�@�33A  BJ�Bp=q@W�@�
A��B���B	�                                    Bxy�@  T          A*ff@\��@���A�
BJ��Bm�@\��@G�A�B�ffB�                                    Bxy��  T          A)p�@\��@�  A (�BD��Bpz�@\��@#�
A�HB�#�B=q                                    Bxy�"�  
Z          A(Q�@e�@�  @��BAz�Bl�R@e�@'�A��B��B��                                    Bxy�12  �          A(��@e@�Q�@�(�BAz�Bl�R@e@(Q�A��B�{B��                                    Bxy�?�  T          A)G�@o\)@�{@��BA�RBg�@o\)@#33A��B�G�B	33                                    Bxy�N~  �          A)G�@e@�p�A
=BJ�RBf��@e@��A\)B��A��\                                    Bxy�]$  "          A'�@mp�@�\)A{BKz�B_�@mp�@�
AG�B�G�A�ff                                    Bxy�k�  
�          A(z�@z�H@�Q�A�BH=qBZ=q@z�H@
=A��B���A��                                    Bxy�zp  �          A(��@s�
@�ffA
=BK��B\Q�@s�
@�A{B��3A�ff                                    BxyΉ  
�          A(��@l(�@��HA�BP33B]\)@l(�?��A\)B���A׮                                    BxyΗ�  T          A(��@i��@�33AQ�B`�BN=q@i��?��A��B���A}��                                    BxyΦb  "          A(��@z=q@��HA�HBL�BW\)@z=q?���AG�B�  A�                                      Bxyε  "          A(z�@g
=@�@���BB�Bj��@g
=@'
=Az�B��HB=q                                    Bxy�î  �          A(��@qG�@�ffA��BQ=qBX
=@qG�?��
A=qB�33Aɮ                                    Bxy��T  
Z          A)p�@qG�@��@�ffBC�Bd�@qG�@!�A��B��Bz�                                    Bxy���  �          A((�@k�@���@�Q�B>z�Bj��@k�@2�\A
=BB�\                                    Bxy��  "          A((�@u�@�33@���B:��Bgff@u�@8Q�A��B{z�B�\                                    Bxy��F  
�          A(  @~�R@�p�@�  B6�Bd@~�R@@��A�
Bv�\BG�                                    Bxy��  �          A(  @_\)@��@�G�B7Bu��@_\)@N{AB|ffB*��                                    Bxy��  
�          A(��@���@XQ�AG�BP��A�  @���=�A�
Bi�
?��R                                    Bxy�*8  T          A)@�p�@�
=@�\)B:�HB\33@�p�@0  A�Bw�RB�R                                    Bxy�8�  !          A*�\@�=q@�\)@��B<��B_
=@�=q@/\)A
=Bz33B�H                                    Bxy�G�  
�          A+
=@�=q@�  @�p�B?  BTQ�@�=q@   A�BxA�(�                                    Bxy�V*  �          A+�@��@�G�@��B;33B]��@��@3�
A\)Bx(�B{                                    Bxy�d�  T          A+\)@�  @�
=A (�BB(�BJG�@�  @p�A
=Bx33Aљ�                                    Bxy�sv  "          A*�R@��H@�
=A ��BDp�BN\)@��H@��A�
B{p�A�p�                                    Bxyς  "          A*�H@��@��
AG�BD\)BHQ�@��@
=A\)ByQ�A�G�                                    Bxyϐ�  �          A*�H@��@���A (�BB�BO�@��@�
A�By��Aߙ�                                    Bxyϟh  �          A*�\@�\)@��@�p�B?�HBL�R@�\)@
=A{Bv�A�z�                                    BxyϮ  �          A*�\@�=q@���@�G�B<  BL  @�=q@\)A��Br��A�R                                    Bxyϼ�  �          A*=q@�(�@��@���B<�\BH�@�(�@�HAQ�BrffA܏\                                    Bxy��Z  
�          A*=q@��\@���@��\B=�\BI��@��\@��A��Bs�A�G�                                    Bxy��   T          A*�\@�{@�p�@��
B>(�BO�R@�{@   A�Bu��A�\)                                    Bxy��  
�          A*{@�G�@�{@�z�B?��BS�H@�G�@!G�AffBxQ�A�                                    Bxy��L  
(          A*=q@�z�@���A  BK=qBO�H@�z�@�\Ap�B���A�\)                                    Bxy��  �          A*ff@��H@�=qA�HBQ33BLff@��H?�\A�RB��=A���                                    Bxy��  T          A)��@��@�=qAp�BOG�BJz�@��?�ffAp�B�aHA�p�                                    Bxy�#>  T          A)�@��
@�p�A=qBH=qBR��@��
@p�AQ�BQ�Aᙚ                                    Bxy�1�  
�          A*�\@��
@�=qA��BE�RBU�@��
@�A��B}��A�G�                                    Bxy�@�  T          A*=q@�=q@�
=A�RBH��BU\)@�=q@��A��B��A�=q                                    Bxy�O0  �          A*ff@�  @���A��BE��BZ�@�  @p�A��BffA�=q                                    Bxy�]�  
�          A)@z�H@�33A�BG��B\(�@z�H@=qA��B���A�33                                    Bxy�l|  "          A)@w�@���A��BO�RBW
=@w�@�\A�RB�W
A�                                      Bxy�{"  
�          A*=q@z=q@�
=A��BV(�BOG�@z=q?�
=A(�B��)A�Q�                                    BxyЉ�  
�          A*�R@{�@��A
ffBX��BLQ�@{�?ǮA��B�aHA���                                    BxyИn  
�          A+�@z�H@��A	��BU\)BQ  @z�H?�\A�B��RA�=q                                    BxyЧ  T          A+\)@qG�@�
=A��BT{BX�R@qG�?�Q�AG�B�\)A�{                                    Bxyе�  
�          A+33@aG�@�(�A33BY�B^  @aG�?�A�HB��A�p�                                    Bxy��`  T          A*�\@~�R@�(�A33BQ�BP@~�R?�z�A33B�  A��H                                    Bxy��  "          A)��@u�@���A
=BS  BU�@u�?�
=A
=B�G�A֏\                                    Bxy��  
�          A*{@w
=@�ffA�HBQ��BU@w
=?��RA33B���A�Q�                                    Bxy��R  �          A*�\@vff@�p�A�
BS(�BU=q@vff?���A  B�8RA֣�                                    Bxy���  �          A*=q@vff@�=qAQ�BT��BS\)@vff?���A  B��A��                                    Bxy��  
�          A*=q@|��@�Q�ABOG�BT\)@|��@AffB�W
A���                                    Bxy�D  "          A*�R@p  @��\A
=BQQ�B[z�@p  @��A  B�B�A�\)                                    Bxy�*�  
�          A*�R@r�\@�G�A\)BQ�
BY��@r�\@ffA(�B�.A��
                                    Bxy�9�  
�          A*�R@p��@�Q�A�
BR�HBY@p��@�
AQ�B���A�Q�                                    Bxy�H6  	.          A*�R@o\)@��\A33BQz�B[��@o\)@	��A(�B�8RA�G�                                    Bxy�V�  (          A+�@~{@��A	p�BTBO��@~{?�{A��B���A��H                                    Bxy�e�  
�          A+�@{�@�  A�BQ(�BT�R@{�@�A(�B���A�33                                    Bxy�t(  "          A+\)@{�@��
A��BS�RBR
=@{�?���AQ�B���A���                                    Bxyт�  
�          A+\)@��H@��RA33BYp�BD
=@��H?�G�Az�B��fA�Q�                                    Bxyёt  (          A*�H@~{@�G�A
�RBY\)BI�@~{?�{Az�B��3A�z�                                    BxyѠ  �          A*�H@���@�Ap�B_�\B?33@���?�(�A��B���A�33                                    BxyѮ�  
�          A*{@~{@���AG�B`�B?�@~{?���A��B�{A�(�                                    Bxyѽf  
�          A*=q@~�R@���A{Bb�B<(�@~�R?���A��B�aHAp��                                    Bxy��  �          A*{@�  @���A�BbQ�B;�\@�  ?�=qA��B��Aq��                                    Bxy�ڲ  "          A*{@���@�Q�A�HBd�B0Q�@���?Q�A(�B�(�A3\)                                    Bxy��X  "          A*�\@�{@\)A33BdffB.�@�{?L��A(�B���A-                                    Bxy���  "          A*=q@�(�@y��A  Bf��B-=q@�(�?333Az�B���A{                                    Bxy��  "          A*�R@��
@i��A{Bk��B&\)@��
>�G�A�B�G�@��                                    Bxy�J  �          A+�
@~{@�(�A
�RBX(�BK�\@~{?�  A��B�=qA���                                    Bxy�#�  "          A+�
@h��@��\A(�BI  Bh{@h��@2�\A\)B�� B�H                                    Bxy�2�  T          A,Q�@s�
@�{A��BJ  B`�R@s�
@)��A�B��B33                                    Bxy�A<  
�          A,Q�@�  @���AG�BJBXQ�@�  @\)A�HB�\A�G�                                    Bxy�O�  
Z          A,z�@�G�@��\A	BS�BM�@�G�?��RA��B���Aх                                    Bxy�^�  T          A-p�@���@�\)A	p�BQ��BP�H@���@��A�B���A��H                                    Bxy�m.  "          A-@���@��A��BP=qBR��@���@�RA�B���A���                                    Bxy�{�  
�          A-�@���@�33A	�BP(�BTG�@���@G�Ap�B�A��                                    BxyҊz  T          A-�@~{@�Q�A�BM�BX��@~{@{A��B���A��\                                    Bxyҙ   
�          A-G�@~{@�ffA�BN
=BW@~{@=qAz�B�8RA��\                                    Bxyҧ�  �          A-@�G�@�{A�BM�\BU�@�G�@=qAz�B���A�Q�                                    BxyҶl  �          A-G�@y��@���A�BM��BZ��@y��@   A��B�aHBp�                                    Bxy��  
�          A-�@w
=@��A�BN�B[�\@w
=@{A��B���B��                                    Bxy�Ӹ  �          A+�
@u@���Az�BRQ�BW��@u@��AQ�B��A�p�                                    Bxy��^  T          A+�@{�@��A  BZz�BK
=@{�?�G�AG�B��3A��R                                    Bxy��  
�          A+�@w�@�ffAffBN
=BZ�@w�@�RA33B�Q�B��                                    Bxy���  T          A+�
@u�@�{A�HBN�B[��@u�@{A�B���Bff                                    Bxy�P  T          A,  @r�\@�=qA�BLB_
=@r�\@'
=A\)B�B�B
ff                                    Bxy��  
�          A+�
@p��@��HAp�BL=qB`=q@p��@)��A
=B�#�B                                    Bxy�+�  "          A*�\@\)@�p�A	G�BVG�BK�H@\)?���A
=B���A�(�                                    Bxy�:B  "          A)@qG�@��HA
=qBZ�BP(�@qG�?�{A�B�B�Aх                                    Bxy�H�  �          A)�@p��@�(�A
�\BZQ�BQ�@p��?��A  B�Q�A�{                                    Bxy�W�  �          A)@s�
@�\)A��BW33BRQ�@s�
@�A33B�A�=q                                    Bxy�f4  T          A*ff@tz�@�G�Ap�B`��BG�@tz�?��
A�B��\A�z�                                    Bxy�t�  
�          A*�R@u�@��A��B^33BJff@u�?�A��B��HA�ff                                    BxyӃ�  
�          A*ff@u@��
A��B^BH��@u?У�A��B��HA�{                                    BxyӒ&  
�          A*=q@q�@�G�Ap�Ba
=BH�H@q�?�ffA�B���A�Q�                                    BxyӠ�  �          A)G�@dz�@�Q�A  Bi(�BH{@dz�?��RA{B��A��                                    Bxyӯr  �          A(Q�@s�
@���A33B^��BHff@s�
?�\)A�HB��qA��\                                    BxyӾ  
�          A'�
@xQ�@�(�A
=BV(�BM�@xQ�@G�Az�B��3Aۮ                                    Bxy�̾  �          A&�H@dz�@��
A
{B_��BQp�@dz�?�(�A{B�L�A�ff                                    Bxy��d  T          A&=q@~�R@���A�BR  BK�\@~�R@�Ap�B�u�A��
                                    Bxy��
  �          A&{@|(�@��
Az�BS�BL(�@|(�@�A{B�.A޸R                                    Bxy���  �          A&{@qG�@�G�Az�B]p�BI33@qG�?�Q�AQ�B��A�G�                                    Bxy�V  �          A&�\@~�R@��\A��BT(�BJ  @~�R@�\A=qB�{A�33                                    Bxy��  �          A%��@{�@�A33BQ�
BM�@{�@�A�B��A�                                    Bxy�$�  �          A%p�@xQ�@���A33B[�BE�R@xQ�?�p�A�HB�� A�=q                                    Bxy�3H  
(          A%��@^�R@�=qA��Bk(�BE�
@^�R?�
=A�RB�W
A��                                    Bxy�A�  
(          A%G�@c33@���A(�Bh33BE��@c33?��AB�{A�33                                    Bxy�P�  
�          A%p�@vff@��HA
=BZ��BH  @vff?�ffA�HB�� A���                                    Bxy�_:  
�          A%G�@q�@�\)A(�B]�RBGz�@q�?�Q�A�B�A��H                                    Bxy�m�  T          A%�@S�
@�z�AG�Bl�BMp�@S�
?��\A�RB��qA�=q                                    Bxy�|�  �          A$(�@�33@���A�\BS{B(33@�33?��RAz�Bw=qA��
                                    Bxyԋ,  T          A#�@�G�@���AB[�\B,�R@�G�?��A�HB�A�ff                                    Bxyԙ�  "          A#�@���@~{A  Ba��B2Q�@���?�(�A��B�W
A�\)                                    BxyԨx  "          A#
=@�  @���AG�B[G�B.z�@�  ?�{A�\B�
=A�Q�                                    BxyԷ  T          A"�H@��\@��\A(�BX�B-=q@��\?�A��B}��A�G�                                    Bxy���  T          A#\)@~{@���A  Ba�B6Q�@~{?�=qA�B���A�                                    Bxy��j  
(          A#\)@x��@��A\)B`=qB;@x��?���A�B�  A�p�                                    Bxy��  �          A#�
@w�@xQ�A
ffBf��B4ff@w�?���A�RB���A�ff                                    Bxy��  �          A#\)@q�@���A	G�Bd��B;z�@q�?�ffA=qB��A��                                    Bxy� \  "          A!G�@<(�@`��A33B{��BHQ�@<(�?333Ap�B�AW33                                    Bxy�  
�          A�R@%@B�\A�B�p�BF33@%>aG�A�B��\@�p�                                    Bxy��  T          A�@'�@J�HAG�B�8RBI�@'�>�33AB�33@�33                                    Bxy�,N  "          A�@,��@>{A�B�p�B>�
@,��>#�
Ap�B�u�@R�\                                    Bxy�:�  
�          A�@'�@?\)A�\B�\BB��@'�>.{A{B�� @g�                                    Bxy�I�  �          A�
@(��@:=qA�HB��RB?33@(��=�Q�A{B�\)?�33                                    Bxy�X@  "          A�@ ��@G�A{B��{BL�R@ ��>��
A=qB��\@�p�                                    Bxy�f�  T          A (�@#33@<��A�B�{BD@#33>\)A�HB�k�@B�\                                    Bxy�u�  �          A (�@'�@>{A
=B�p�BBz�@'�>#�
A�\B���@b�\                                    BxyՄ2  �          A Q�@)��@C�
A�\B�Q�BD(�@)��>��AffB��@�(�                                    BxyՒ�  "          A ��@3�
@FffA=qB���B?G�@3�
>��
AffB�Q�@�p�                                    Bxyա~  
�          A!G�@U@Z�HA{Bv�HB6�
@U?333A�
B�z�A>ff                                    Bxyհ$  
Z          A!��@K�@QG�AQ�B|��B7=q@K�?�AG�B�ǮA33                                    Bxyվ�  
�          A!@Mp�@J�HA��B~�B2ff@Mp�>�
=AG�B���@�                                    Bxy��p  �          A!G�@O\)@L(�A�
B|�B2=q@O\)>�AQ�B�  AG�                                    Bxy��  �          A!��@S33@c33A�Bup�B<=q@S33?Y��A(�B��{Ag�                                    Bxy��  T          A!p�@QG�@a�A�Bv
=B<�\@QG�?W
=A(�B��)Af�\                                    Bxy��b  "          A!��@N�R@\(�A�HBx��B;=q@N�R?=p�A��B��3AO�                                    Bxy�  
�          A!��@P  @W�A
=ByQ�B8�@P  ?0��Az�B��A>=q                                    Bxy��  �          A!p�@U�@U�A�RBx�RB4�@U�?&ffA  B��3A2{                                    Bxy�%T  T          A ��@a�@QG�AG�Bu��B+\)@a�?!G�AffB�=qA"=q                                    Bxy�3�  T          A!�@\��@W
=Ap�BuB1{@\��?8Q�A�HB�\A<                                      Bxy�B�  �          A!�@^{@W�Ap�BuQ�B0�R@^{?:�HA�HB���A>�H                                    Bxy�QF  
�          A ��@U�@Tz�A=qBx�\B3�@U�?+�A�B��A6�H                                    Bxy�_�  "          A ��@`��@U�A�Bt��B.33@`��?8Q�AffB�L�A8��                                    Bxy�n�  �          A Q�@a�@X��A�
Br��B/ff@a�?L��Ap�B���ALz�                                    Bxy�}8  
�          A Q�@fff@Z�HA33Bq33B.Q�@fff?Y��A��B��qAS�                                    Bxy֋�  "          A (�@r�\@a�A��Bk\)B+�@r�\?}p�A\)B��Ak\)                                    Bxy֚�  
�          A Q�@n{@^�RA
{Bm�HB,p�@n{?n{A(�B�
=AaG�                                    Bxy֩*  "          A   @tz�@e�Az�BiB,p�@tz�?��A
=B�B�Ax��                                    Bxyַ�  
�          A�
@w�@hQ�A�Bg��B,�R@w�?���A=qB�p�A���                                    Bxy��v  �          A\)@�G�@e�A�Bd��B&Q�@�G�?�\)A��B�G�Aw�
                                    Bxy��  �          A�
@���@\��A�Bg�B"�@���?xQ�Ap�B��AY�                                    Bxy���  �          A   @~{@b�\A\)BgG�B&�@~{?���AB�aHAq�                                    Bxy��h  "          A Q�@�z�@fffA=qBc(�B$(�@�z�?�33A��B�8RAyG�                                    Bxy�  
�          A   @���@aG�A33Bf�B$�@���?��Ap�B���Alz�                                    Bxy��  �          A   @\)@l��A{Bc��B+ff@\)?�G�A�B�u�A�Q�                                    Bxy�Z  T          A (�@��\@dz�AffBd��B$�
@��\?��A��B��HAy�                                    Bxy�-   T          A   @��@[�A33Bf��B��@��?�  A�B�AZ{                                    Bxy�;�  �          A Q�@�z�@Z=qA\)Bf��B�
@�z�?xQ�A�B�ǮAS33                                    Bxy�JL  
�          A\)@���@_\)A��Bc�B p�@���?��A�B��Al(�                                    Bxy�X�  �          A33@|��@H��A	�Bn
=B  @|��?333AB�k�A Q�                                    Bxy�g�  
�          A�H@���@W
=A�HBh�\B�R@���?s33Az�B��
AU��                                    Bxy�v>  
�          A�@��\@\��A�HBf�\B �H@��\?��A��B��Aep�                                    Bxyׄ�  �          A�@��@c�
A�B]��B��@��?��HA�Bz��Av�\                                    Bxyד�  
�          A�@�ff@_\)A�B]��Bff@�ff?�33ABzQ�Ag\)                                    Bxyע0  "          A�
@���@Z�HA��B`G�Bff@���?���AffB|�AZ�R                                    Bxyװ�  T          A�
@���@QG�A=qBZ�B	p�@���?xQ�A\)Br��A7�
                                    Bxy׿|  �          A�
@��@HQ�A Q�BU�\A�  @��?aG�A��Bl�A�                                    Bxy��"  
Z          A�@�\)@Y��@�G�BMG�B(�@�\)?���A�\Be�AN�R                                    Bxy���  
(          A\)@��@a�@��BR\)B
=@��?�ffA��Bm\)Am��                                    Bxy��n  
�          A33@�ff@XQ�Ap�BY��B�H@�ff?�{A33BsAT��                                    Bxy��  "          A\)@�  @W�A�B^�Bp�@�  ?���AG�Byz�AV{                                    Bxy��  
�          A33@���@AG�A�BcG�B��@���?5AG�Bz=qA�R                                    Bxy�`  �          A�
@�ff@9��A��Baz�A��@�ff?��AQ�Bvz�@��                                    Bxy�&  �          A (�@�@G�A�\Bd�HB��@�?L��A
=B}�A#\)                                    Bxy�4�  "          A Q�@�=q@G�ABb  B	=q@�=q?O\)A{By�A!G�                                    Bxy�CR  �          A Q�@�z�@P��AQ�Bi  B��@�z�?k�A�B��AG�                                    Bxy�Q�  �          A z�@r�\@Y��A
=qBm��B'ff@r�\?��\A�B��fApQ�                                    Bxy�`�  "          A Q�@qG�@^{A	��Bl�HB*Q�@qG�?���A\)B�ǮA�(�                                    Bxy�oD  "          A   @{�@Z�HAz�BjG�B$Q�@{�?�=qA�B��)Aw
=                                    Bxy�}�  "          A�@�  @QG�A�\Be�HB33@�  ?xQ�A�B�#�AM��                                    Bxy،�  	�          A�@�=q@QG�A{Bdp�Bz�@�=q?z�HA
=B~�\AMG�                                    Bxy؛6  
�          A�R@�  @O\)ABe�RBG�@�  ?uA�\B�
ALz�                                    Bxyة�  �          A�\@���@S�
A��Bc33B33@���?�ffAB}�
A\                                      Bxyظ�  
�          A�\@�=q@UAQ�BbG�B�H@�=q?���AB}�Ac�
                                    Bxy��(  �          A�H@�ff@W�Ap�Bd��B  @�ff?�{A�HB��AmG�                                    Bxy���  T          A�@|��@UA��Bk=qB �@|��?��AB��qAk33                                    Bxy��t  �          A�
@�@^�RA{Bd�B{@�?�(�A�
B�W
A�                                    Bxy��  T          A�@�Q�@`��A��Ba�\B{@�Q�?��\A�HB~�A�33                                    Bxy��  �          A�@���@`  Az�B`��B��@���?��\AffB}
=A��                                    Bxy�f  
Z          A33@��\@Z=qA��Ba�B��@��\?�Q�A{B|Au                                    Bxy�  �          A33@�33@W�Az�Ba�\B{@�33?�z�A�B|Q�An=q                                    Bxy�-�  T          A33@��
@W�AffBf��B=q@��
?���A�B�#�Au�                                    Bxy�<X  �          A33@���@R�\A�Bj�BQ�@���?��A��B��Ah                                      Bxy�J�  
�          A
=@�ff@X��A��BdffB��@�ff?�
=A�HB�HAz{                                    Bxy�Y�  "          A�R@�@Y��A�BdG�B��@�?���A�\B��A�
                                    Bxy�hJ  �          A
=@��\@^{A�Bez�B!p�@��\?�G�A�B�{A���                                    Bxy�v�  
�          A33@z�H@`  A33BhffB'�@z�H?��\A��B�\A�                                    Bxyم�  "          A�@l(�@^{A	p�Bm�
B,�H@l(�?��HA
=B��A��                                    Bxyٔ<  
�          A
=@p��@aG�A  BkG�B,ff@p��?��
AB��A�z�                                    Bxy٢�  T          A�\@p  @`  A�Bk=qB,
=@p  ?��
AG�B��
A�z�                                    Bxyٱ�  �          A{@}p�@X��A=qBh�\B"=q@}p�?���A�B�u�A��\                                    Bxy��.  �          A�
@�{@eA�BaB"Q�@�{?�z�A33B~��A���                                    Bxy���  T          A   @�z�@_\)AQ�B_z�B�H@�z�?�=qA{Bz�RA��H                                    Bxy��z  �          A�@�{@[�A(�B_ffB��@�{?��
A��By�
A���                                    Bxy��   T          A�@�z�@UA��Ba�B
=@�z�?�Q�A{B{z�Ar{                                    Bxy���  "          A�
@���@S�
A(�B^�B33@���?�
=A�Bw��Ah                                      Bxy�	l  "          A�@�33@S�
AG�Bb�
B
=@�33?�z�A=qB|p�Ao�                                    Bxy�  �          A�R@�Q�@I��A{BfB��@�Q�?�  AffB33AR�H                                    Bxy�&�  	�          A�@�ff@G�A��Bg�B��@�ff?�  A�B��AV{                                    Bxy�5^  T          A��@z=q@R�\ABjQ�B �@z=q?�33A�\B��fA�\)                                    Bxy�D  �          A��@�z�@G
=A��Bh=qB�R@�z�?}p�A�B�p�AX                                      Bxy�R�  "          A��@�Q�@=p�A�Bh=qB�@�Q�?Y��A��B~��A4                                      Bxy�aP  �          A�@�\)@>�RAp�Bh�B�@�\)?^�RA�B�A8��                                    Bxy�o�  
�          A=q@�G�@J=qAG�Be��B��@�G�?�ffA��B}��A\(�                                    Bxy�~�  T          A��@~{@l(�A�Bb�B+�R@~{?˅A�B��\A��                                    BxyڍB  "          Ap�@��H@aG�A�Bb�B#{@��H?�Q�AG�Bz�A��H                                    Bxyڛ�  �          AG�@�\)@Z=qA
=Ba�Bz�@�\)?��AQ�B|��A���                                    Bxyڪ�  
'          A�\@�Q�@e�A�Bd�B'  @�Q�?�(�A
=B��A�p�                                    Bxyڹ4  "          A\)@�G�@tz�A  B`
=B-�\@�G�?�p�A�HB{A���                                    Bxy���  �          A�H@��\@UA�\Bg��B��@��\?�p�A\)B��A��
                                    Bxy�ր  
�          A z�@�G�@uAp�B`�B.
=@�G�?޸RA(�B�HA�{                                    Bxy��&  
�          A ��@��@p  A�Ba��B)��@��?�33AQ�Bz�A��H                                    Bxy���  
�          A ��@~�R@l��A
=Bd��B+��@~�R?˅AG�B�k�A��                                    Bxy�r  �          A!�@z=q@[�A	G�Bk
=B$��@z=q?�ffAffB�z�A���                                    Bxy�  T          A ��@xQ�@A�Az�Br��B��@xQ�?\(�A  B�� AH��                                    Bxy��  "          A!@z=q@L��A(�Bp  B�@z=q?��AQ�B�ǮAn{                                    Bxy�.d  �          A!p�@x��@P��A�BoG�B��@x��?�{A  B�A
=                                    Bxy�=
  "          A!@x��@QG�A�
Bo=qB G�@x��?�\)AQ�B�ǮA���                                    Bxy�K�  
Z          A!��@w�@P��A�
Bo�B �@w�?�\)A(�B�  A�33                                    Bxy�ZV  "          A!p�@j�H@C33A{Bv�B�@j�H?c�
A��B�  AX��                                    Bxy�h�  �          A!G�@e@>{A�HByBz�@e?L��A{B�.AH��                                    Bxy�w�  
Z          A!G�@g�@0��A�
B|z�Bff@g�?
=A=qB�L�A�                                    BxyۆH  T          A!�@qG�@Mp�A(�Br
=B!�@qG�?�=qAQ�B��A�                                      Bxy۔�  T          A ��@w
=@]p�A
{Bl(�B'ff@w
=?���A33B�{A��\                                    Bxyۣ�  "          A Q�@s33@VffA
ffBn�\B%��@s33?�  A
=B���A�                                    Bxy۲:  T          A�@p  @[�A	G�BmffB)�
@p  ?���AffB��HA�ff                                    Bxy���  �          A�@z=q@P  A��Bm�B��@z=q?�
=AG�B��=A��R                                    Bxy�φ  
�          A�H@u@:=qA
�HBs�HB��@u?Tz�A�B�ffAB�R                                    Bxy��,  "          A�H@�  @A�A	�Bn��B�\@�  ?xQ�A��B�  AZ�H                                    Bxy���  �          A�@��
@FffA��BkffB�
@��
?��AQ�B�z�Ae�                                    Bxy��x  �          A�
@�(�@N{A(�Bi��B��@�(�?�
=AQ�B�{A~ff                                    Bxy�
  
(          A33@vff@\��A  Bj�HB'p�@vff?�z�A�B�aHA��R                                    Bxy��  	�          A   @r�\@VffA	�Bn\)B%@r�\?��A�\B�A�{                                    Bxy�'j  "          A�H@n�R@\(�Az�Bm�B*��@n�R?�33Ap�B��3A�ff                                    Bxy�6  
Z          A{@j�H@c33A33BkG�B033@j�H?��A��B�k�A��                                    Bxy�D�  �          AG�@g�@l��ABh�
B6�@g�?��HA�B���A�                                      Bxy�S\  
a          A33@u�@qG�A�Be=qB2  @u�?�\A(�B�ǮA�(�                                    Bxy�b  M          A ��@���@�  Az�B^(�B3�@���@G�A�B}33A��                                    Bxy�p�  �          A   @�
=@��
A�BW\)B1=q@�
=@(�A��BvffA�G�                                    Bxy�N  
Z          A!G�@=p�?�(�A  B�ǮA���@=p��uAz�B��)C��                                    Bxy܍�  
�          A"{@녾��
AB��qC�\@��!�A��B��RC�H                                    Bxyܜ�  �          A Q�@�\?�RA�RB�p�Ar�H@�\����A�B���C���                                    Bxyܫ@  �          A ��@
=�8Q�A(�B��RC���@
=��A�B��C�xR                                    Bxyܹ�  
�          A�
@#�
?��AQ�B�#�A�@#�
�!G�A�B�Q�C��                                    Bxy�Ȍ  �          A
=@9��?�
=A��B��B�R@9���L��A  B���C��                                    Bxy��2  �          A�R@QG�@1�A33B��RB!G�@QG�?5Ap�B��=AD(�                                    Bxy���  
�          A�H@aG�@�A�B�\)B  @aG�>��RAz�B�L�@��                                    Bxy��~  
�          A�H@]p�@�A��B�ffB �
@]p�=�G�A��B�=q?���                                    Bxy�$  "          A�H@X��@z�A��B��)A��@X��    AG�B��                                        Bxy��  
�          Aff@E?���A33B��B @E�\)AffB��\C���                                    Bxy� p  
�          A�H@G�?�A�B���A��@G��#�
A�RB�G�C�~�                                    Bxy�/  "          A�R@^{@�RA
=B��B33@^{>�ffAQ�B���@�\                                    Bxy�=�  
�          A�H@e�@5�A�Bz��B\)@e�?O\)A�B���AM�                                    Bxy�Lb  
Z          A�R@^�R@#33A�\B�W
B  @^�R?�A(�B�\)A	�                                    Bxy�[  �          A\)@`  @7
=A�B|  B��@`  ?W
=Az�B���AX                                      Bxy�i�  �          A�@s�
@7�A(�Bu�RB@s�
?aG�A�HB���AO
=                                    Bxy�xT  
�          A   @z�H@L��A	�BnffB  @z�H?��RAB�z�A�                                      Bxy݆�  T          A Q�@�
=@N{A  Bh�Bz�@�
=?��
A  B��A���                                    Bxyݕ�  
�          A z�@��@L(�A  Bh
=B��@��?�G�A�
B(�A�Q�                                    BxyݤF  
S          A�
@�@P  A�Bg�HB��@�?�=qA�BA�p�                                    Bxyݲ�  
Z          A\)@��R@-p�A	BoQ�B
=@��R?E�A�
B��A&=q                                    Bxy���  
�          A
=@���@�A
ffBq��A���@���>��HA�B�Ǯ@��                                    Bxy��8  T          A33@�@��A
�HBsQ�A�
=@�?�\A(�B��H@�
=                                    Bxy���  �          A�R@��@(Q�A
{Br(�Bz�@��?333A  B�Q�Aff                                    Bxy��  
Z          A33@�33@*�HA
�\Br{B{@�33?:�HAz�B�k�A"ff                                    Bxy��*  T          A�H@�  @C�
A��Bn  B��@�  ?��A(�B�k�A33                                    Bxy�
�  
�          A�R@{�@(��A�Bv{Bz�@{�?333Ap�B��{A ��                                    Bxy�v  
�          Aff@��
?��AB~p�A�@��
����A�
B��=C�4{                                    Bxy�(  �          A�\@�{?�Q�AB}��A�Q�@�{���HA\)B���C���                                    Bxy�6�  "          A�\@��
?��
AG�B|(�A�33@��
���A(�B�ǮC��
                                    Bxy�Eh  "          A�\@���@	��Az�By��A�\)@���>aG�Az�B���@G
=                                    Bxy�T  
�          A�\@�Q�@ ��A\)Bu�A��@�Q�?
=A��B��A                                    Bxy�b�  �          A{@��@G�A
�\Bu(�A�z�@��>�p�A
=B��H@�=q                                    Bxy�qZ  �          A�@{�@�
A��B}{Aݙ�@{�>\)A��B�
=@�\                                    Bxyހ   T          A=q@���@(Q�AQ�Bn  A�G�@���?B�\A=qB�HA"{                                    Bxyގ�  �          A�R@�@P  A  Bap�B��@�?�A  Bx{A�=q                                    BxyޝL  �          A�H@�33@QG�A�RB]p�B�R@�33?�(�A
�RBs��A�p�                                    Bxyޫ�  
�          A
=@���@(�A33Bt33A�  @���?��AQ�B�=q@�Q�                                    Bxy޺�  �          A
=@���@�AG�Bz�RA�(�@���>W
=AG�B�  @@��                                    Bxy��>  "          A�@�Q�@   A
=qBp�HA��H@�Q�?�RA�B��Az�                                    Bxy���  �          A33@�{@<(�A�\Bf�RB
=@�{?���Ap�Bz��A_
=                                    Bxy��  �          A�H@�33@0��A�
Bj�HB��@�33?k�A{B}�\A>{                                    Bxy��0  
�          A�R@�
=@-p�A	�Bn�B�H@�
=?Y��A
=B���A6ff                                    Bxy��  �          A�H@�Q�@&ffA	G�Bo(�A�33@�Q�?@  A
=B�L�A
=                                    Bxy�|  "          A
=@~{@=qA��Bx��A�z�@~{?�A��B��@�R                                    Bxy�!"  �          A�\@�G�@+�AQ�Bm
=B {@�G�?Y��A=qB(�A2=q                                    Bxy�/�  
�          A{@��
@$z�A	Br=qA��@��
?8Q�A33B��A                                    Bxy�>n  �          A{@�Q�@ ��A
�RBuffB (�@�Q�?&ffA  B�\)A\)                                    Bxy�M  
�          A=q@�(�@ ��A	�Bs
=A�=q@�(�?(��A\)B���A��                                    Bxy�[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxy�j`              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxy�y  
�          A�R@�\)@!G�A	Bq
=A�(�@�\)?.{A
=B��
A                                    Bxy߇�  T          A�R@�@p�A
ffBr��A�{@�?�RA�B��=A33                                    BxyߖR  T          A�R@�{@   A
=qBr(�A��H@�{?(��A�B�aHA(�                                    Bxyߤ�  �          A�R@�
=@'
=A	G�Bo��A��
@�
=?G�A
=B���A(Q�                                    Bxy߳�  �          A�\@��@'
=AQ�Bm�A���@��?L��A{B~��A((�                                    Bxy��D  
�          A{@~�R@+�A
{Bs�RB�@~�R?Y��A  B�W
A@z�                                    Bxy���  
�          Ap�@���@=p�A�Bm�
Bff@���?�z�AffB���A�Q�                                    Bxy�ߐ  �          A{@�p�@5�A�
Bm\)B�R@�p�?��\A=qB�� A]�                                    Bxy��6  
�          Aff@}p�@(Q�A
=Bu=qB
=@}p�?J=qA��B���A4                                      Bxy���  �          A=q@���@.{A	��Bq�RB�@���?fffA�B�ffAIp�                                    Bxy��  T          A{@\)@)��A
{Bs�
Bz�@\)?Tz�A�
B�8RA<z�                                    Bxy�(  T          A��@tz�@%�A33Bx=qB�@tz�?@  A��B�aHA1                                    Bxy�(�  T          Ap�@{�@#33A
ffBvG�B  @{�?:�HA�
B�\A)�                                    Bxy�7t  "          AG�@~{@"�\A	�BuQ�B�@~{?=p�A\)B�� A(                                      Bxy�F  T          A�@�z�@&ffA(�Bp�\B �@�z�?O\)AB��A1�                                    Bxy�T�  T          A��@��H@%�AQ�Bq�\B  @��H?L��AB���A0z�                                    Bxy�cf  T          A��@���@)��A�
Bo�\B�@���?^�RA��B�ǮA=G�                                    Bxy�r  
�          A��@�(�@2�\A33Bm�B��@�(�?�G�Ap�B���A\��                                    Bxy���  �          A��@�G�@0��A�Bp�B	p�@�G�?}p�A�B�A[�                                    Bxy��X  �          A=q@}p�@!G�A33Bv�RB�H@}p�?5Az�B�A!                                    Bxy���  
�          A{@s33@z�A��B|��A���@s33>��HA��B�L�@�=q                                    Bxyତ  �          A@^�R?��HA��B���A���@^�R����A\)B�ffC���                                    Bxy�J  �          AG�@`  ?�p�A��B���A�z�@`  �!G�A�\B��=C��                                    Bxy���  �          Ap�@g�?�\)A(�B�p�A��H@g��k�AffB��RC�8R                                    Bxy�ؖ  "          Ap�@o\)?�{A\)B�{A��\@o\)�aG�A��B�8RC�N                                    Bxy��<  �          A��@{�?�A��BA�@{�    Az�B���C��R                                    Bxy���  �          A��@�z�@ffA
�HBw��A��@�z�>��RA�HB�.@�=q                                    Bxy��  �          A{@���@A	p�Bq��A�33@���?�A=qB�#�@��H                                    Bxy�.  �          A=q@�
=@�HA	Br{A�R@�
=?&ffA�RB��
A�
                                    Bxy�!�  �          A{@�  @ffA	BrG�A�  @�  ?z�AffB��{@��H                                    Bxy�0z  �          A�@�p�@z�A
=qBtG�A���@�p�?��A�HB��@�                                      Bxy�?   �          A=q@�G�@�RA�RBi�RA�R@�G�?@  A  Bx��A�                                    Bxy�M�  
�          A@j�H@��A{B��)A�p�@j�H>��
A{B�  @���                                    Bxy�\l  T          A��@^{?��RA33B��
A�z�@^{>.{A�RB�z�@.�R                                    Bxy�k  
�          A��@j�H@�A�B���A�{@j�H>�p�A{B�
=@��                                    Bxy�y�  T          A��@x��@�\A(�B{  A�@x��?   A��B��@��H                                    Bxy�^  �          A��@w�@Q�A�By�A�ff@w�?��AQ�B�
=Az�                                    Bxy�  T          A(�@mp�@p�A
=B{p�B(�@mp�?0��A  B�� A(                                      Bxyᥪ  T          A�H@�=q@%�A=qBpz�B�R@�=q?^�RA�B�\AAG�                                    Bxy�P  �          A\)@�p�@.�RAG�Blz�B
=@�p�?��
A\)B
=A]�                                    Bxy���  �          A��@�  @XQ�A��Bg(�B �R@�  ?�A��BG�A���                                    Bxy�ќ  
Z          Ap�@��
@8��A
=Bl��B{@��
?�z�A��B�W
A{�
                                    Bxy��B  
�          A{@~{@>{A��Boz�Bp�@~{?�p�A\)B�B�A���                                    Bxy���  T          Aff@z�H@^{AffBhffB&{@z�H?޸RA�RB��A�                                    Bxy���  
�          A��@�  @E�A=qBk�HB�@�  ?�\)AG�B��HA�
=                                    Bxy�4  
�          A��@��@C�
A{Bk{B(�@��?���A�B�G�A�G�                                    Bxy��  "          A=q@���@9��A�HBi�B�R@���?�Q�AG�B}G�Axz�                                    Bxy�)�  �          A{@�  @G�A��Bf�HB=q@�  ?�A��B|
=A�                                    Bxy�8&  �          Ap�@���@J�HAp�Bg�HB�@���?�(�A��B}��A�=q                                    Bxy�F�  T          A�@�ff@*=qA�
Bn�\B\)@�ff?p��A��B�{AJ{                                    Bxy�Ur  	.          A=q@�=q@A
=qBsA�ff@�=q>�33A{B�H@�                                    Bxy�d  �          A�@�{@Q�A
�HBv33A�@�{>ǮA�HB��@��                                    Bxy�r�  T          A=q@�
=@z�A
{BsQ�A�(�@�
=?z�A�RB��f@��                                    Bxy�d  �          A�@�  @!�A�
Bo{A�  @�  ?Q�A�BffA.ff                                    Bxy�
            A@�p�@�RA
�\Bu�A���@�p�>��HA�HB���@ָR                                    Bxy➰  
�          A{@�
=@\)A	G�Bq  A�  @�
=?B�\AffB�� A#�
                                    Bxy�V  �          A�\@���@(�A	��Bp��A�p�@���?5A�\B�{A
=                                    Bxy��  	�          A�H@��@\)A
ffBr�RA�G�@��?�\A�RB�{@�                                    Bxy�ʢ  �          A�@�p�?�AQ�BvA�  @�p���\)A�HB~�C���                                    Bxy��H  �          A   @��?�z�A�\B{��A��R@�녾�33A(�B��C��f                                    Bxy���  "          A33@���?�=qA�Bu  A�(�@��׾��AB{�C��                                    Bxy���  T          A
=@�G�?�\A
�\BrffA��\@�G�=L��Ap�B{�?.{                                    Bxy�:  �          A�H@�(�?���A��ByQ�A��R@�(���z�A�\B(�C�q                                    Bxy��  T          Aff@���?�A��B{�A���@��þ��
A�RB��{C�޸                                    Bxy�"�  "          A�H@��?�
=A�B}
=A��H@������A�B�\)C��                                    Bxy�1,  T          A�R@���?���A�B~{Ag\)@��ÿ#�
A�RB�8RC��q                                    Bxy�?�  �          A�\@�ff?s33A�HB�� AL(�@�ff�L��A
=B��fC���                                    Bxy�Nx  
�          A�R@��R?p��A�HB�ffAI�@��R�O\)A
=B��RC���                                    Bxy�]  "          A�H@�(�?J=qA�
B��
A,��@�(��xQ�A�B�\)C�c�                                    Bxy�k�  
�          A
=@�=q>�
=A��B�k�@�z�@�=q����A�B��C��R                                    Bxy�zj  �          A33@}p�>W
=A�B��@Dz�@}p��ǮA�B�B�C�E                                    Bxy�  "          A"�H@�\)?���A��B{��Ao�@�\)�z�A�B~��C�O\                                    Bxy㗶  �          A"�\@�ff?�=qA�B|AZ�H@�ff�0��A��B~�C��{                                    Bxy�\  �          A"�\@��?�(�Az�Bz��As\)@�����A��B~=qC��H                                    Bxy�  �          A"�H@�  ?�G�A�B|Q�AI�@�  �E�A��B}z�C�&f                                    Bxy�è  �          A"�R@���?^�RA�B|ffA-@��׿fffA��B|33C�XR                                    Bxy��N  �          A"ff@��?fffAz�B{  A1��@�녿^�RAz�B{�C���                                    Bxy���  T          A"�R@�ff?\(�A��B~
=A/�@�ff�h��A��B}��C�8R                                    Bxy��  
�          A"�H@�
=?c�
A��B}�\A3�@�
=�c�
A��B}�\C�b�                                    Bxy��@  �          A"�H@�  ?Y��Ap�B}�A+\)@�  �k�Ap�B|C�7
                                    Bxy��  
�          A"=q@�?Y��Ap�B~z�A-p�@��k�AG�B~�C�)                                    Bxy��  T          A"�H@��?0��AB~{AQ�@����=qA�B|(�C�=q                                    Bxy�*2  �          A"�H@�\)?
=A�B~��@�
=@�\)��
=A��B{�RC��R                                    Bxy�8�  �          A#
=@���?
=qAB}�R@�=q@��׿�p�A��BzffC�j=                                    Bxy�G~  T          A"�R@���?z�AG�B|��@�Q�@�����
=AQ�By�C��
                                    Bxy�V$  �          A"ff@�  ?   Ap�B~
=@���@�  ��G�A(�BzffC�33                                    Bxy�d�  �          A"�\@��>��A�B|@�z�@�녿���A�BxQ�C��=                                    Bxy�sp  �          A"�\@�ff>�\)A�B�R@j=q@�ff��p�A(�By�C�Ф                                    Bxy�  �          A"{@�>��A��B�@Tz�@���  A�By��C���                                    Bxy䐼  �          A!@���>uAp�B�B�@J=q@��Ϳ�G�A�BzG�C���                                    Bxy�b  �          A!�@�=q<#�
A�\B��==�\)@�=q��G�A�
Bz\)C��=                                    Bxy�  �          A"ff@��
��A�\B�  C�Ф@��
���A�By{C�ٚ                                    Bxy伮  �          A"=q@���=#�
A{B��?
=q@��Ϳ�(�A�Bx�
C�XR                                    Bxy��T  �          A!�@��R=uAG�Bz�?8Q�@��R��Q�A�RBw�\C��R                                    Bxy���  �          A!��@��R>�A��B33?��@��R��\)A�\Bx  C��                                    Bxy��  �          A!��@�{=�Q�A�B�\?�{@�{��z�A�RBw��C��                                     Bxy��F  �          A!G�@�G�>�z�A�B��=@z=q@�G���(�A(�B}�C��                                    Bxy��  �          A ��@�
=>8Q�A�
B~�@�@�
=�ǮA��Bwz�C�h�                                    Bxy��  �          A (�@���?��Ap�B�z�Az�@��Ϳ�z�A��B�
=C�5�                                    Bxy�#8  �          A�
@�
=>ǮA��B��3@��\@�
=����A
=B~z�C�                                      Bxy�1�  �          A
=@�p�>���AQ�B��@�\)@�p����A�HB\)C��                                    Bxy�@�  �          A�\@��
?�\A  B��@�  @��
���RA�HB���C��                                    Bxy�O*  �          A�R@�(�?�A  B�Q�@��@�(���A
=B��qC��                                    Bxy�]�  �          A�R@��
?   A(�B��=@�@��
���RA
=B���C���                                    Bxy�lv  �          A�\@���>��HA��B�Ǯ@�{@��׿�G�A�B��3C�J=                                    Bxy�{  �          Aff@���?�AQ�B�ffA ��@��ÿ�
=A\)B�ǮC�޸                                    Bxy��  T          A��@���>�ffA�B�.@�z�@������
A=qB���C�4{                                    Bxy�h  �          AG�@{�>L��A  B�Ǯ@<��@{����
A�B�#�C�U�                                    Bxy�  �          A�@xQ�>���A  B�33@�(�@xQ쿴z�AffB�8RC��                                    Bxy嵴  �          AG�@y��=�G�AQ�B�=q?�z�@y���У�A�B��C���                                    Bxy��Z  �          A��@r�\>��A��B�z�@�\@r�\�˅A�\B�k�C���                                    Bxy��   T          AG�@j�H>�z�A�B�\@�Q�@j�H��(�A  B���C�\                                    Bxy��  �          AG�@\��>�33A33B��@�G�@\�Ϳ�
=Ap�B�W
C��H                                    Bxy��L  T          A��@HQ�>W
=Az�B���@y��@HQ��=qAffB�\C��)                                    Bxy���  �          A��@W��B�\A�HB���C�XR@W����HA�B�C��                                    Bxy��  �          AG�@k�>aG�A��B��@XQ�@k����A�B�{C��                                     Bxy�>  �          A��@c33=�G�AffB��\?�=q@c33��33A  B���C���                                    Bxy�*�  �          AQ�@��
�#�
A��B��C��R@��
��  A
�HBz��C��                                     Bxy�9�  �          A
=@�녾�p�Az�B�#�C�g�@����
A��BwC���                                    Bxy�H0  �          A�R@��\�J=qA	B{�
C���@��\��A��Blz�C�S3                                    Bxy�V�  �          A�H@�ff�:�HA��Bx�C�Q�@�ff��A(�BjffC��
                                    Bxy�e|  �          A�R@�(��+�A	p�Bz�
C���@�(��z�A��Bl��C�
=                                    Bxy�t"  �          Aff@�G��B�\A	B|C�H@�G����A��Bm�C�Y�                                    Bxy��  �          Aff@�33�8Q�A	G�B{�C�K�@�33��Az�BlffC��q                                    Bxy�n  �          A�\@��
�5A	G�Bz�
C�g�@��
�ffAz�BlG�C�ٚ                                    Bxy�  �          A=q@�z���A��Bz��C�R@�z��  Az�Bm  C�p�                                    Bxy殺  �          A�@�Q��A�
Bw��C��q@�Q��
=A�
Bkz�C�w
                                    Bxy�`  �          A@������HA33Bv(�C��=@�����A
=Bj  C��                                    Bxy��  T          A��@��ÿ0��A=qBup�C���@����33A��Bg�\C��=                                    Bxy�ڬ  �          A��@�33�O\)Az�Bz\)C��3@�33���A\)Bj��C�L�                                    Bxy��R  �          A{@��׿���A�HBt��C�XR@����+�A ��Bc��C���                                    Bxy���  *          A{@����{A  BxG�C�ٚ@���/\)A�BfffC���                                    Bxy��  ~          A��@xQ�=��
Az�B�W
?��R@xQ��\)A
{B�\C���                                    Bxy�D  �          A@s�
�\A��B�C�'�@s�
��A	�B|�C��=                                    Bxy�#�  �          Aff@vff��AG�B�z�C�\@vff�\)A��Bz
=C��                                    Bxy�2�  �          A=q@��\�L��A�B��)C���@��\��  A��Bz33C�ff                                    Bxy�A6  �          Ap�@�G�<#�
A	G�B}�R=��
@�G���z�A�RBu\)C�j=                                    Bxy�O�  �          A��@�=q�   A��B|=qC���@�=q�	��A��Bo33C��)                                    Bxy�^�  �          AG�@�p��aG�A33Bw��C�Z�@�p�� ��A�Bg�
C�1�                                    Bxy�m(  �          A�@�=q����A��Bq�C�� @�=q�=p�@�B^�C��\                                    Bxy�{�  �          Aff@��Ϳ�(�A�Bn\)C�8R@����C�
@��
BZ��C�P�                                    Bxy�t  �          A@��\����A  BlffC��q@��\�Z�H@�\)BV{C��R                                    Bxy�  �          AG�@z=q���A	��B��C�f@z=q��ABxffC���                                    Bxy��  �          Az�@u�>�A�B�L�@�G�@u���G�A
=qB�  C���                                    Bxy�f  �          AQ�@Q�?E�AffB��\AR�R@Q녿}p�A{B��
C��                                     Bxy��  �          A�
@fff?�RA  B��RA33@fff��{A33B�8RC�q�                                    Bxy�Ӳ  �          A�@tz�?�A
=qB���@�{@tz῕A	G�B��C�z�                                    Bxy��X  �          Aff@�{�B�\A�B|33C�ٚ@�{�Q�A ��Blz�C�+�                                    Bxy���  �          A�\@����AG�By  C�H�@��:�H@��BdC���                                    Bxy���  �          A@�Q�}p�A{B(�C��@�Q��'�A z�Bl��C�l�                                    Bxy�J  �          A��@tz�\AQ�B�k�C�%@tz��33Az�By\)C���                                    Bxy��  �          AG�@n{��A��B��C�@n{��ffAB~�RC��                                    Bxy�+�  �          A��@U>�p�A
ffB��\@�33@U���A��B�ffC�
                                    Bxy�:<  �          AQ�@E�?��A
{B���A���@E���(�A�B��C��                                    Bxy�H�  �          A��@.{?��
A
�HB��fB@.{=#�
AB���?L��                                    Bxy�W�  �          A@3�
@ffA
ffB���BG�@3�
>�Q�AffB���@�Q�                                    Bxy�f.  �          A��@"�\@�A
=qB�� B*\)@"�\?z�A�HB�z�AO33                                    Bxy�t�  �          A(�@!�@
=A	p�B�\B,=q@!�?�RA{B�L�A]��                                    Bxy�z  �          A��@�?��RAz�B��fB$=q@�>aG�A  B�aH@�{                                    Bxy�   T          Az�@@ffAz�B���B4�@>���Az�B��{A�                                    Bxy��  �          A(�@�R@ffA
�RB�B9�H@�R?
=A\)B�#�Ak�                                    Bxy�l  �          Az�@33?�z�AQ�B�\)B  @33>\)A�B�8R@Y��                                    Bxy�  �          A�@'
=?�33A�B�ffA��@'
=�#�
A�RB��C�%                                    Bxy�̸  �          A  @HQ�?&ffA
�\B�� A;�
@HQ쿊=qA	�B���C�w
                                    Bxy��^  �          A��@U�����A
�HB�C�,�@U�����A�B��
C��                                    Bxy��  �          AQ�@o\)�\)A\)B��fC��H@o\)��RA�HBx  C���                                    Bxy���  �          A��@vff���\A�\B�L�C���@vff�+�A ��Bn��C��
                                    Bxy�P  �          A��@�(�����A�Bz\)C���@�(��,(�@�33Bgz�C�s3                                    Bxy��  �          A�@����ABvp�C�T{@��9��@�Ba��C��f                                    Bxy�$�  �          A��@aG����A	��B�p�C��\@aG���{AffB�B�C��                                    Bxy�3B  �          A�@@  >B�\A��B�G�@j�H@@  ����A
�\B�L�C�/\                                    Bxy�A�  �          A�@4z�?!G�A�\B��AI�@4z῕AB�  C��q                                    Bxy�P�  �          Aff@8��?333A�RB��AY��@8�ÿ���A{B��\C��{                                    Bxy�_4  �          A�R@:�H?�RA�HB�A>=q@:�H��Q�A�B��)C��\                                    Bxy�m�  �          A�\@3�
?�A\)B���A+\)@3�
���
A=qB��RC��H                                    Bxy�|�  �          A�\@4z�?(�A\)B�B�AD��@4zῙ��AffB���C��                                     Bxy�&  �          A=q@1�?   A33B���A$Q�@1녿��A�B���C�e                                    Bxy��  �          A=q@2�\?   A33B��HA#\)@2�\���A�B��3C�ff                                    Bxy�r  �          Aff@/\)?   A�B�z�A%�@/\)����A=qB�33C�%                                    Bxy�  �          A{@(��>��HA�B�A(Q�@(�ÿ�=qAffB�B�C��3                                    Bxy�ž  �          A{@)��?
=qA�B�k�A6ff@)�����AffB�L�C�
=                                    Bxy��d  �          A=q@.{>���A�B��fAG�@.{��
=A�B�C�/\                                    Bxy��
  T          A=q@2�\>\A\)B�\@�=q@2�\��Q�A��B��fC�^�                                    Bxy��  T          A{@,��>�(�A�B�\A��@,�Ϳ�33A�B��C�T{                                    Bxy� V  �          A��@*�H?(�A
=B��fAL  @*�H��(�A{B�W
C���                                    Bxy��  �          A��@,��?
=A�HB��\AC\)@,�Ϳ��RAB��)C���                                    Bxy��  �          AG�@4z�>�ffA{B�\)A�@4z`\)Az�B��3C�f                                    Bxy�,H  �          Ap�@6ff?   A{B��HA�H@6ff����A��B���C���                                    Bxy�:�  �          A�@5>�G�A�B��A�@5����AQ�B�p�C��                                    Bxy�I�  �          Ap�@<��>��A��B��)@��
@<�Ϳ�z�A  B�\C�33                                    Bxy�X:  �          A�@p��#�
A�B�k�C��f@p����Az�B�8RC�J=                                    Bxy�f�  T          A�@z὏\)AQ�B�.C�0�@z��z�A�B�\)C�9�                                    Bxy�u�  �          A��@%�>��A�HB���@�  @%���=qA��B�
=C�E                                    Bxy�,  T          Az�@0  >uAB�ff@�  @0  �˅A�B�  C��                                    Bxy��  �          A��@,�ͼ#�
A{B�33C��q@,�Ϳ�A
=B���C�ٚ                                    Bxy�x  �          AQ�@)��<��
A{B�>���@)����A33B�u�C��3                                    Bxy�  �          AQ�@0��=�\)A��B�G�?�ff@0�׿�  A
�RB���C��\                                    Bxy��  �          Az�@1G�=#�
AB�8R?c�
@1G����A
�RB�W
C���                                    Bxy��j  �          Ap�@333�W
=A�\B�  C��@333�33A
�RB�ffC���                                    Bxy��  �          A��@3�
��\)A{B��
C�O\@3�
��z�A
�RB�33C��\                                    Bxy��  �          A��@?\)��=qA��B�=qC�g�@?\)�A��B��RC���                                    Bxy��\  T          A��@E�����A  B���C��3@E����A�
B�L�C��q                                    Bxy�  �          A(�@HQ쾸Q�A\)B�8RC��)@HQ��
=qA33B�z�C���                                    Bxy��  �          A��@N�R���RA33B��C�E@N�R��A33B��HC�aH                                    Bxy�%N  �          A  @J=q���
A
=B���C�{@J=q�Q�A�HB�Q�C��                                    Bxy�3�  �          A�@G
=����A
�RB�Q�C��@G
=���A�\B���C���                                    Bxy�B�  �          A�@B�\����A33B�Q�C��f@B�\�	��A
=B�u�C�b�                                    Bxy�Q@  �          A�
@A녾�=qA�B��=C�xR@A��A�B�C���                                    Bxy�_�  �          A  @Mp��W
=A
�RB�G�C�&f@Mp��G�A�HB��=C��                                    Bxy�n�  �          A�@[���Q�A��B�\C�
=@[����Az�B�HC��)                                    Bxy�}2  �          A=q@��?��A�B�Av{@�Ϳ��\A  B���C��                                    Bxy��  �          Aff?�p�?�
=A�\B�z�B	(�?�p��8Q�A\)B�L�C���                                    Bxy�~  �          A(�?�{?��A��B��\B233?�{���A{B�B�C��                                    Bxy�$  �          A=q?˅?�=qA	��B��qB3{?˅��\)A�
B�
=C�
                                    Bxy��  �          Aff?�p�?�33A
ffB�B�BT?�p��k�A��B���C��3                                    Bxy��p  �          A��?���?�  A
�RB�k�B8?�����RA�B�p�C�K�                                    Bxy��  �          A�R?���?�Q�A
=B���Bc��?��;B�\A��B��{C��q                                    Bxy��  �          A�\?\(�?�\A
�RB�z�B�\)?\(���A��B��)C�H                                    Bxy��b  T          A��?���?��A
=qB��qB2�H?��ͿG�A
�HB�W
C�AH                                    Bxy�  �          A�?��H?.{Az�B��qA��R?��H��z�A�B���C��
                                    Bxy��  T          A�\?��?fffA
�HB�A���?���z�HA
�RB��{C���                                    Bxy�T  T          A�H?�(�?s33A\)B��=A��H?�(��p��A\)B���C���                                    Bxy�,�  �          A�@�?�
=A
ffB��{A��H@녿5A33B�33C�k�                                    Bxy�;�  �          A
=@	��?���A	B�z�A�ff@	���@  A
ffB��=C�l�                                    Bxy�JF  �          A\)@G�?��\A	B�ǮA��H@G��\(�A	�B��\C��3                                    Bxy�X�  �          A�@ff?��A	p�B��A��H@ff�W
=A	B��C�+�                                    Bxy�g�  �          AQ�@ ��?p��A	��B�\)A�Q�@ �׿p��A	��B�\)C��R                                    Bxy�v8  �          A��@0  �\A	�B�Q�C��@0  ��RAG�B��C�q�                                    Bxy��  �          AG�@.{�uA	B�  C�@ @.{�3�
A
=B�
=C�f                                    Bxy쓄  �          A��@4z�=p�Az�B�=qC��@4z��$z�A�RB��)C���                                    Bxy�*  �          Ap�@8�þ�p�A	��B��=C�^�@8����RA��B��=C�+�                                    Bxy��  �          A
=@-p��ǮA(�B�B�C��H@-p���\A\)B�u�C��                                    Bxy�v  �          A�@>{�^�RA	G�B�Q�C���@>{�.�RA�HB
=C��
                                    Bxy��  �          A�@2�\�\Az�B��C���@2�\�Vff@��Bup�C��                                    Bxy���  �          Aff@8����A�B�#�C���@8�����@���Bb��C���                                    Bxy��h  �          A@(Q���A�B���C���@(Q����R@�(�BcC���                                    Bxy��  �          A��@+��2�\A�HB�ffC���@+�����@�BY��C�P�                                    Bxy��  �          A�@*=q�(��A(�B�#�C��R@*=q����@�
=B]�
C���                                    Bxy�Z  �          A��@"�\�.�RA  B�ffC�q�@"�\���@�{B]=qC��q                                    Bxy�&   �          A(�@�R�0��A\)B�\)C��@�R��Q�@���B\��C�b�                                    Bxy�4�  �          A�@��>�RA�B�\)C��
@���ff@�Q�BW�\C��                                    Bxy�CL  �          A  @%��9��AB��HC��)@%���(�@��BW��C���                                    Bxy�Q�  �          A�
@*�H�/\)A{B���C�#�@*�H��\)@�=qBZ(�C�e                                    Bxy�`�  �          Az�@7
=���A\)B�#�C��)@7
=��p�@�
=B_��C�4{                                    Bxy�o>  �          A��@.�R�/\)A�RB�G�C�p�@.�R��  @�33BY�\C��H                                    Bxy�}�  �          AQ�@6ff�%A�\B�z�C���@6ff��33@��
B[=qC��                                     Bxy팊  �          AQ�@'
=�&ffA
=B���C��=@'
=���
@���B]�C�g�                                    Bxy�0  �          A  @Mp��Q�A=qB�\)C�/\@Mp��z=q@�ffB_ffC���                                    Bxy���  �          A�@U��A ��BffC���@U�xQ�@��
B\�
C�W
                                    Bxy��|  �          A  @U�ffAp�B�C���@U�w�@��B]��C�k�                                    Bxy��"  �          A�@W��
�HA ��B}��C���@W��|(�@��HB[�C�Ff                                    Bxy���  �          A(�@O\)�z�A�B
=C�4{@O\)���\@��HBZ�C�1�                                    Bxy��n  �          A�
@E�\)A ��B~�C���@E��  @��BXz�C��                                    Bxy��  �          A\)@>{�,��A (�B}�\C��f@>{��ff@�{BUffC��                                    Bxy��  �          A�@Mp��(�A Q�B}G�C�c�@Mp���ff@�Q�BW�C���                                    Bxy�`  �          A�@C�
�'�A (�B}�C��
@C�
��(�@�ffBU��C�z�                                    Bxy�  �          A�R@O\)��\@��B~Q�C�g�@O\)����@�  BY��C�U�                                    Bxy�-�  �          A=q@J�H��@�{B}  C�AH@J�H��{@�p�BW  C��3                                    Bxy�<R  �          A@C�
�#�
@�p�B}  C�H@C�
���@��
BU�\C��=                                    Bxy�J�  �          A��@?\)�,(�@�(�B{��C�H@?\)��@ᙚBSp�C���                                    Bxy�Y�  �          Ap�@>{�2�\@��HBz{C�h�@>{����@߮BP�C���                                    Bxy�hD  �          AG�@B�\�1G�@��Bx�C��
@B�\��  @޸RBP(�C��                                    Bxy�v�  �          A�@B�\�9��@�  Bv
=C�0�@B�\���
@��
BL��C��3                                    Bxy  �          A��@8���H��@�Bs�RC�O\@8�����H@׮BHp�C�n                                    Bxy�6  �          AQ�@5��[�@�=qBn\)C��H@5����H@��BA�RC���                                    Bxy��  �          A(�@0���tz�@�(�BeC��@0����{@�G�B7�C�o\                                    Bxy  �          A�@.�R���@�Ba��C��R@.�R��p�@ƸRB2��C�޸                                    Bxy��(              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxy���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxy��t   \          Az�@.{����@�=qBa�C�f@.{��(�@�p�B2p�C��                                    Bxy��  �          A�
@(Q����@�{BRp�C��)@(Q��Å@��B!��C���                                    Bxy���  
�          A��@ ����ff@��HBKC�l�@ ����@�\)B�HC���                                    Bxy�	f  �          A�@#�
����@�ffBZ�
C��@#�
��\)@�ffB)�HC���                                    Bxy�  �          A��@4z��{�@�Bb��C��3@4z����@�
=B3��C�q�                                    Bxy�&�  �          AG�@>{�E@��RBs�RC��@>{���\@�Q�BG��C���                                    Bxy�5X  �          A�@H���Q�@�33B�k�C��@H���{�@��
BZ��C�K�                                    Bxy�C�  �          A
�H@S33�6ff@�  Bo\)C��3@S33����@�33BF\)C��3                                    Bxy�R�  T          A
�H@HQ��
=@��B|�RC�� @HQ���z�@޸RBU33C���                                    Bxy�aJ  T          A
=q@J=q����@�z�B���C���@J=q�Z=q@�G�Be�
C�h�                                    Bxy�o�  �          A	�@O\)���R@�(�B�z�C��@O\)�E@�33Bj��C�*=                                    Bxy�~�  T          A��@W���=q@�ffB�\)C�u�@W��XQ�@��HB`(�C�xR                                    Bxy�<  �          A	�@aG���{@�{Bz�C�n@aG��J�H@�z�Ba�C��)                                    Bxy��  �          A
{@aG���ff@���B�u�C��@aG��HQ�@�\)Bc��C�(�                                    Bxy愈  �          A	@[���=q@���B�\)C�l�@[��J�H@�\)Bdz�C���                                    Bxy�.  �          A��@W
=��=q@���B�#�C�  @W
=�;�@���Bj{C�z�                                    Bxy���  �          A��@O\)�h��@��HB�z�C�#�@O\)�2�\@�(�Bo��C���                                    Bxy��z  T          AQ�@Mp����
@�=qB�aHC��@Mp��:=q@�\Bm��C��=                                    Bxy��   �          AQ�@S�
����@���B���C�ٚ@S�
�=p�@��BjQ�C��                                    Bxy���  �          A�
@P�׿��\@���B���C�K�@P���8��@�G�Bl��C�4{                                    Bxy�l  �          A\)@B�\��\)@���B��=C��@B�\�N�R@�ffBiffC���                                    Bxy�  T          A�@R�\���@��B���C�33@R�\�:=q@�  Bk(�C�=q                                    Bxy��  �          A�@L�Ϳ��@���B�33C��R@L���:�H@��Bl�C��=                                    Bxy�.^  T          A33@G�����@���B���C�Q�@G��>�R@��BmQ�C�%                                    Bxy�=  T          A33@G
=����@���B�(�C���@G
=�=p�@���Bm�HC�=q                                    Bxy�K�  �          A
=@Dz῀  @�G�B���C��@Dz��9��@陚Bo�C�XR                                    Bxy�ZP  �          A�H@?\)��  @�=qB�(�C���@?\)�9��@�\Bq�\C��f                                    Bxy�h�  �          A�\@>{�n{@��B�z�C�K�@>{�6ff@�\BrC�"�                                    Bxy�w�  �          Aff@<(��s33@��B��)C��@<(��7
=@�\Bs  C�޸                                    Bxy��B  �          A�\@=p����@���B��C�"�@=p��>�R@�G�BpQ�C�h�                                    Bxy��  
�          A
=@@  ����@�G�B��C�  @@  �G
=@�  Bl�C���                                    Bxy�  �          A33@<�Ϳ�
=@��\B�ǮC��@<���Fff@���Bn�C���                                    Bxy�4  �          A�H@B�\��Q�@���B�ffC�P�@B�\�Fff@�
=Bk�
C�33                                    Bxy���  �          A�\@E��(�@�
=B�u�C�AH@E�G�@�p�Bi��C�XR                                    Bxy�π  �          Aff@C�
���
@�
=B��\C���@C�
�L(�@���Bi(�C��                                    Bxy��&  �          A�R@A녿��R@�Q�B�33C��R@A��J�H@�{Bj�C��                                     Bxy���  �          A@?\)����@�
=B�  C�� @?\)�C33@�Bl�C�33                                    Bxy��r  �          A��@<�Ϳ���@�
=B�u�C�� @<���C�
@�Bmz�C���                                    Bxy�
  �          A�@=p���@�ffB�C�+�@=p��U@�\Bg�C�˅                                    Bxy��  �          Ap�@<�Ϳ��R@��B�\)C���@<���X��@��Be�C�|)                                    Bxy�'d  �          Az�@.�R���R@�
=B�u�C��@.�R�K�@�z�Bn\)C�\)                                    Bxy�6
  �          A(�@,(����@�\)B�� C�� @,(��E@�p�Bq
=C��\                                    Bxy�D�  T          Az�@1녿��@�B�B�C�!H@1��QG�@�\Bk(�C�.                                    Bxy�SV  �          A��@9����=q@�B��C���@9���_\)@�ffBcp�C��R                                    Bxy�a�  �          Ap�@>{��@�=qB���C�N@>{�l��@��HB]
=C�Y�                                    Bxy�p�  �          A��@E����\@�B��3C��=@E��L��@��BfC��R                                    Bxy�H  �          Az�@@  ���R@�B��)C��H@@  �J�H@���Bh�RC��                                    Bxy��  T          A��@0�׿�p�@�{B��RC��q@0���[�@�G�Bh  C�c�                                    Bxy�  �          A��@3�
����@�B�.C�W
@3�
�Y��@�G�Bg�\C���                                    Bxy�:  �          A��@3�
����@�B��)C�0�@3�
�S�
@�=qBi��C�,�                                    Bxy��  T          A\)@G
=����@��B�� C���@G
=�C�
@޸RBg�C��
                                    Bxy�Ȇ  �          A�@Mp����@�B�8RC��
@Mp��AG�@�{Bf=qC�b�                                    Bxy��,  �          A(�@J�H����@�B�33C��=@J�H�AG�@�Q�Bg��C�33                                    Bxy���  �          A�@I����z�@�  B�ǮC���@I���E@�Be��C��                                     Bxy��x  �          A�@>�R�u@�33B�L�C�
@>�R�;�@�\Bm�
C���                                    Bxy�  �          A33@8Q�W
=@�B���C��H@8Q��4z�@�Bq��C��=                                    Bxy��  �          A�@<(��@  @�33B��=C���@<(��/\)@�(�BrffC�w
                                    Bxy� j  �          A(�@<(��n{@�33B��HC�+�@<(��:�H@�\Bn�C���                                    Bxy�/  �          A�@
=q�<(�@�ffBz�C��@
=q��=q@�z�BI�C�\                                    Bxy�=�  �          A��?�=q�Z�H@���B|{C�]q?�=q����@�
=BA�C�P�                                    Bxy�L\  T          A��?�33�hQ�@�\Bu�HC�@ ?�33���R@�33B;��C�l�                                    Bxy�[  �          A��?��H�]p�@��
ByffC��?��H���@�p�B?z�C��                                    Bxy�i�  �          A  @0  �Q�@��BQ�C��@0  ��G�@ϮBO�\C�XR                                    Bxy�xN  �          A�
@/\)��@�p�B�B�C���@/\)���@�Q�BP�HC�t{                                    Bxy��  �          A�
@.{�33@�B���C��@.{��
=@���BQ�HC�c�                                    Bxy�  �          A�@)���  @�RB�
=C�ٚ@)����@�=qBT  C�.                                    Bxy�@  
�          Az�@+���@�G�B�L�C��{@+����\@�BWffC���                                    Bxy��  �          Az�@HQ��  @��B�z�C�0�@HQ��_\)@ڏ\B^(�C��                                    Bxy���  �          Az�@<(���  @�G�B�8RC��f@<(��n�R@���B[�HC��                                    Bxy��2  �          A(�@@  ��(�@�B�8RC��=@@  �^�R@ۅB`�C�b�                                    Bxy���  �          A(�@Fff��{@�=qB��C�+�@Fff�H��@�\)Bf�RC�U�                                    Bxy��~  �          Az�@Dzῥ�@�=qB�z�C���@Dz��Tz�@�BcffC�c�                                    Bxy��$  �          AG�@P�׿z�H@��HB��C���@P���A�@���Bf��C��=                                    Bxy�
�  �          A�@I�����\@�(�B�\C��@I���E@ᙚBg�C�˅                                    Bxy�p  �          Az�@G
=���@��HB�{C�K�@G
=�I��@߮Bf�RC�N                                    Bxy�(  �          A�@B�\���
@�B���C���@B�\�Vff@���Bd��C��                                    Bxy�6�  �          A=q@?\)��G�@���B���C��@?\)�G�@�Bk��C��q                                    Bxy�Eb  �          Aff@<(����@���B�z�C�H�@<(��J�H@�ffBlG�C�l�                                    Bxy�T  �          A@<(��xQ�@�Q�B�u�C��f@<(��E@�Bm{C��=                                    Bxy�b�  T          Az�@>�R�Y��@�B�
=C�{@>�R�=p�@�z�Bn{C��)                                    Bxy�qT  �          A�@?\)�Tz�@�
=B�=qC�Ff@?\)�=p�@�Bn��C���                                    Bxy��  �          A  @=p��O\)@��B�W
C�N@=p��;�@��
BnC��                                     Bxy�  �          A�@>�R�O\)@�(�B���C�k�@>�R�:�H@��HBn(�C��f                                    Bxy�F  �          Az�@<�ͿTz�@�{B��\C�*=@<���=p�@�z�Bn�C�o\                                    Bxy��  �          A��@>{��  @�{B���C��{@>{�HQ�@��HBj�C���                                    Bxy�  �          A��@@  �J=q@�{B�  C���@@  �;�@���BnG�C��R                                    Bxy��8  �          A\)@C�
�0��@��HB��C���@C�
�4z�@�\BnG�C��=                                    Bxy���  �          A�
@C�
�!G�@�(�B�� C�(�@C�
�1�@�(�Bo�RC�޸                                    Bxy��  �          AQ�@AG���@�{B��C��H@AG��z�@�\BzG�C�:�                                    Bxy��*  �          A��@HQ���@�\)B��\C�H�@HQ��&ff@�G�BsG�C�#�                                    Bxy��  �          A{@I���O\)@�
=B�\)C��3@I���>�R@�p�BkG�C�B�                                    Bxy�v  +          A�@I���}p�@�{B��\C�G�@I���J=q@�\BgG�C�u�                                    Bxy�!  }          A��@HQ쿃�
@�B�#�C��)@HQ��K�@߮BeC�E                                    Bxy�/�  �          Ap�@Dz᾽p�@��B�z�C���@Dz��%@陚BtC��                                    Bxy�>h  �          A�@Fff�E�@�B��C�f@Fff�:�H@�Q�BjC�\)                                    Bxy�M  �          A33@@  ���H@�B�C�ff@@  ��Q�@�\)BQ
=C�ff                                    Bxy�[�  �          A
=@<(���@�=qB}�RC���@<(���
=@�(�BLz�C�t{                                    Bxy�jZ  �          Az�@;���Q�@�  B�ǮC�5�@;���G�@ӅBS�RC��{                                    Bxy�y   �          Aff@@���	��@���B\)C�<)@@����Q�@ҏ\BNffC���                                    Bxy�  �          A�
@C33���@�B{Q�C�  @C33���@У�BH�HC�)                                    Bxy��L  �          A�
@C�
��
=@�p�B�z�C���@C�
��33@أ�BS�C�`                                     Bxy���  �          A�
@G����H@�ffB�p�C��@G��y��@ۅBW�HC�W
                                    Bxy���  �          A�@HQ���H@�ffB�G�C��R@HQ��y��@�33BW�\C�e                                    Bxy��>  
�          A33@G��ٙ�@�p�B�=qC��R@G��x��@�=qBWffC�c�                                    Bxy���  �          A\)@O\)�ٙ�@�(�B�ǮC��@O\)�xQ�@���BU=qC��f                                    Bxy�ߊ  T          A\)@J�H���
@�z�B��C�T{@J�H�~{@أ�BT�C�O\                                    Bxy��0  �          A=q@Dz��33@�(�B��HC��)@Dz��vff@ٙ�BX\)C�N                                    Bxy���  �          A�@C33���
@�  B�� C��=@C33�qG�@�{B\Q�C�y�                                    Bxy�|  �          A  @@  ����@�G�B�ǮC��3@@  �w
=@޸RB[��C���                                    Bxy�"  �          A�@=p���33@���B��)C�e@=p��y��@�p�BZ��C���                                    Bxy�(�  �          A\)@?\)��33@�  B�z�C���@?\)�x��@���BZG�C���                                    Bxy�7n  �          A
=@7
=��@�Q�B��
C�� @7
=�z�H@���B[�\C�
=                                    Bxy�F  �          A�H@<(���@�
=B���C�/\@<(��z�H@ۅBYC�q�                                    Bxy�T�  T          A�\@<(���Q�@�{B�ffC��@<(��{�@ڏ\BY
=C�ff                                    Bxy�c`  �          Aff@9����p�@�{B��{C���@9���~{@��BX�\C��                                    Bxy�r  �          Aff@?\)��\@���B��C���@?\)��Q�@�  BU�
C�aH                                    Bxy���  �          Aff@@�׿���@�(�B��\C�q�@@������@�
=BTp�C�N                                    Bxy��R  �          A�R@E���
=@�33B��C���@E�����@���BP�C�G�                                    Bxy���  
�          A�\@?\)��
=@��
B�  C��3@?\)���@�p�BQ�HC��3                                    Bxy���  �          A�\@K��G�@�Q�B}\)C��=@K���
=@���BL  C��H                                    Bxy��D  �          Aff@N{��@�{By�C��\@N{���@��BGG�C�8R                                    Bxy���  �          A=q@U��Q�@���Bw�C���@U�����@�z�BFQ�C��                                    Bxy�ؐ  �          Aff@S33�@�By
=C���@S33����@�BG��C�ٚ                                    Bxy��6  �          A=q@P���Q�@�Byz�C�l�@P����=q@��BGffC���                                    Bxy���  �          Aff@P���Q�@�{By�\C�s3@P�����\@�p�BGffC���                                    Bxy��  �          A=q@S�
�Q�@��Bx�C���@S�
���\@�(�BF�C��\                                    Bxy�(  �          A=q@S�
���@��Bw��C���@S�
���H@�(�BE�
C��\                                    Bxy�!�  �          A=q@\(���@陚Br�C�B�@\(���{@�\)B?�HC��f                                    Bxy�0t  �          A�R@P  �	��@�RBy��C�H�@P  ���
@��BF��C�W
                                    Bxy�?  �          Aff@E��@�G�B�C�\)@E��G�@���BL  C��f                                    Bxy�M�  �          A�@G��Q�@�
=B|�HC��R@G����@�BH�RC��                                    Bxy�\f  T          A{@E��   @���B�\C���@E���Q�@У�BL�C��=                                    Bxy�k  �          A=q@E���\)@�=qB��C�\)@E����@�33BO�C�B�                                    Bxy�y�  �          A�@>{��
=@�\B���C�xR@>{��
=@ҏ\BO��C���                                    Bxy��X  �          A@.�R��Q�@��B��
C�K�@.�R��Q�@��BS{C�L�                                    Bxy���  �          A@)�����
@�\)B�.C��@)����z�@أ�BX(�C�K�                                    Bxy���  �          A�@,(����H@�\)B�#�C��=@,(����\@�G�BX��C��R                                    Bxy��J  �          AG�@*=q�˅@�\)B�\)C���@*=q�~{@�=qB\  C��                                    Bxy���  �          A�@ �׿��
@�(�B��{C�xR@ ���n�R@ᙚBe�C��)                                    Bxy�і  �          A{@�R��Q�@�(�B���C���@�R�xQ�@�  Bb�C�O\                                    Bxy��<  �          Aff@�H��=q@�z�B�ǮC�z�@�H����@�
=B`(�C���                                    Bxy���  �          A�\@�R��\)@�=qB���C��=@�R����@��BW�C��                                    Bxy���  �          A�\@'
=�   @��B�
=C�H�@'
=��(�@�{BR�RC�aH                                    Bxy�.  �          A�R@.{��@��B�\C�� @.{��G�@љ�BLG�C�w
                                    Bxy��  T          A�R@333��H@�=qB��C���@333���@�(�BD��C�P�                                    Bxy�)z  �          Aff@333�ff@�=qB�p�C��@333��p�@���BFffC�z�                                    Bxy�8   �          A@0  ��R@�=qB��HC�|)@0  ��=q@�{BI�C���                                    Bxy�F�  �          A�@*�H�	��@�(�B��=C��3@*�H����@�Q�BLz�C�N                                    Bxy�Ul  �          A@%��@�\B�B�C��)@%���R@�z�BG��C�b�                                    Bxy�d  �          A��@'
=��@�B��C�}q@'
=��  @�G�BE
=C�b�                                    Bxy�r�  �          AQ�@%��
@�  B�aHC�"�@%��z�@ʏ\BH{C��{                                    Bxy��^  T          A�@!��z�@�B�.C�^�@!���@�ffBN�C��q                                    Bxy��  �          A\)@'
=� ��@��B��RC�+�@'
=���
@�{BN�RC�ff                                    Bxy���  �          A\)@)����Q�@���B��)C��\@)�����@�
=BO�HC��\                                    Bxy��P  �          A{@*�H��=q@�\)B�(�C��
@*�H��{@�ffBQ�\C�E                                    Bxy���  +          A ��@0�׿(�@���B��HC���@0���C�
@���Bl��C��q                                    Bxy�ʜ  }          A z�@5�!G�@�\)B��C��
@5�E�@�33Bj=qC�W
                                    Bxy��B  �          A   @7
=�E�@�{B���C�w
@7
=�L��@�Q�BfffC��                                    Bxy���  �          A   @:�H�s33@�(�B���C�H@:�H�Vff@�z�B`��C���                                    Bxy���  �          @��@>�R���@�\B�p�C�^�@>�R�[�@�=qB]Q�C�}q                                    Bxy�4  �          A z�@>�R���\@��HB�ffC�w
@>�R�i��@ϮBX(�C���                                    Bxy��  �          A ��@.�R��\@�33B�G�C�z�@.�R��(�@ʏ\BO�HC��3                                    Bxy�"�  �          A (�@&ff�p�@�  B��{C���@&ff����@��HBE�RC��)                                    Bxy�1&  �          A z�@�H��
@�G�B��3C��@�H��(�@��HBEffC��                                    Bxy�?�  �          A=q@��R@�z�B�u�C��H@���\@��
BC
=C���                                    Bxy�Nr  �          Aff@����@��B�C�XR@�����@��BDC�>�                                    Bxy�]  �          A�@   �ff@�{B���C��@   ���@�G�BJ�C���                                    Bxy�k�  �          A@/\)��=q@���B�
=C�&f@/\)��\)@��HBN�C�u�                                    Bxy�zd  �          A ��@#33�
=q@�=qB��
C���@#33����@�z�BG=qC��q                                    Bxy��
  �          A ��@5��@��B�33C���@5��@ƸRBKz�C�q                                    Bxy���  �          A ��@
=���H@�B�.C�(�@
=��(�@�=qBO(�C�(�                                    Bxy��V  �          A ��@(����@�p�B��HC�%@(����@ʏ\BO��C���                                    Bxy���  �          A ��@#33����@��B�#�C�J=@#33��  @��HBO�HC���                                    Bxy�â  �          A z�@(�ÿ���@�B��C���@(�����@�G�BN=qC��R                                    Bxy��H  �          A ��@"�\�(�@�\B��qC��)@"�\���\@��
BEC���                                    Bxy���  �          @�\)@��R@�RB�\)C���@��=q@��B?Q�C���                                    Bxy��  �          A z�@ff�?\)@�\Bv�
C�
@ff��Q�@��B0�C�\                                    Bxy��:  �          A ��@G��7
=@�B|  C�7
@G���p�@��B5��C��)                                    Bxy��  �          AG�@�����@�B�L�C�U�@����33@���BB
=C�y�                                    Bxy��  �          AG�@���\)@�(�B��C�u�@�����@�z�BF
=C��{                                    Bxy�*,  �          A��@
=q��
=@�B�C�\@
=q��{@���BQffC���                                    Bxy�8�  �          A@
=�У�@��B��{C��@
=���R@ӅBZ�C�O\                                    Bxy�Gx  �          A@�
�\@��
B��C�` @�
���H@ӅBZffC��                                     Bxy�V  �          Ap�@
=��Q�@�33B�C�Y�@
=��Q�@ӅB[p�C�AH                                    Bxy�d�  �          Ap�@ �׿�@��B�33C�K�@ ���~�R@ҏ\BY��C�"�                                    Bxy�sj  �          AG�@(����@��HB�C��@(��xQ�@���B]C�q                                    Bxy��  �          A�@��33@��HB��C���@�~�R@ӅB\=qC�8R                                    Bxy���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxy��\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxy��   �          AG�@
�H��  @��HB���C���@
�H���\@�
=BT��C�XR                                    Bxy���  �          AG�@���(�@�\B��RC�� @�����@�
=BU(�C�s3                                    Bxy��N  �          AG�@z��\)@�Q�B��3C���@z���p�@�33BOQ�C�޸                                    Bxy���  �          A�@	������@���B���C�޸@	����Q�@��HBN�C��q                                    Bxy��  
�          A�@33�Q�@�\)B��=C��q@33���@�\)BH�C�!H                                    Bxy��@  �          A�@��z�@�  B��
C��@����@��BE�C��3                                    Bxy��  �          A�@�
��@�33B�  C�˅@�
����@��B?p�C��)                                    Bxy��  �          A�@=q�%@�  B�
C�z�@=q����@��\B9(�C��3                                    Bxy�#2  �          Ap�@p��3�
@�  B�C�R@p���\)@�  B5z�C�p�                                    Bxy�1�  �          A�@��/\)@�\B��C�E@���ff@��HB8(�C�`                                     Bxy�@~  �          A@��1�@��B~�RC���@���
=@���B5p�C��=                                    Bxy�O$  �          Ap�@33�5@�
=B|�\C��=@33��Q�@�ffB3{C��3                                    Bxy�]�  �          AG�@��>{@��
Bw��C�AH@����@��B-�C��=                                    Bxy�lp  �          @��R@*=q��@ᙚB|��C�/\@*=q����@��RB9  C���                                    Bxy�{  �          @��@7
=�У�@�\B�(�C�"�@7
=��33@�  BH��C�n                                    Bxy���  �          @��\@8Q���@��
B��RC���@8Q���G�@��BJ�C���                                    Bxy��b  �          @��H@8Q쿾�R@�(�B��C�Z�@8Q��\)@��HBL33C��                                    Bxy��  �          @��\@6ff���R@�(�B�k�C�.@6ff��  @��HBLQ�C���                                    Bxy���  �          @���@5���  @�B�u�C�  @5���Q�@��BK�C���                                    Bxy��T              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxy���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxy��   �          @�@+�����@�RB��=C��3@+���\)@�
=BCffC�s3                                    Bxy��F  �          @���@,�Ϳ�(�@�
=B�(�C��@,������@�=qBH�C�*=                                    Bxy���  �          @���@%��(�@�Q�B���C�@ @%��G�@�33BJz�C��
                                    Bxy��  �          @��@'����@�  B��C��R@'����@��BH{C���                                    Bxy�8  �          @�@*=q��\)@�\)B���C�xR@*=q��p�@���BE�\C��                                     Bxy�*�  �          @��@/\)��
=@�
=B���C�=q@/\)��  @�=qBHC�g�                                    Bxy�9�  �          @�ff@.{��{@�\)B�\C��3@.{��@���BD��C�Ǯ                                    Bxy�H*  �          @�\)@,(�����@�  B��qC��@,(�����@��BB��C�^�                                    Bxy�V�  �          @�
=@,(���(�@�\)B�� C��=@,(���G�@�
=BB�C�W
                                    Bxy�ev  T          @��R@0  ��z�@�
=B�.C��
@0  ��\)@�
=BBffC���                                    Bxy�t  �          @�33@,�Ϳ�G�@��B��C�|)@,����=q@�
=BF{C�H                                    Bxy���  �          @�z�@*=q���@�{B�8RC�Z�@*=q��ff@�ffBC��C�l�                                    Bxy��h  �          @��@*�H���@���B��
C�O\@*�H��ff@���BB�RC�t{                                    Bxy��  �          @���@'
=� ��@�B���C�:�@'
=���\@�(�B@��C�޸                                    Bxy���  T          @��R@ ����
@�{B�33C��{@ �����@���B:33C��=                                    Bxy��Z  �          @��@#�
���R@�RB���C�\@#�
���\@�p�BA�HC���                                    Bxy��   �          @�z�@�R�@�{B�ffC��
@�R��p�@�33B?��C��                                    Bxy�ڦ  �          @�z�@ff�
=@��B��C�u�@ff���@�
=B9�C���                                    Bxy��L  �          @�@\)�ff@�\)B�z�C���@\)��ff@�(�B?p�C���                                    Bxy���  �          @�@#33��33@��B��qC���@#33��G�@�\)BC��C��                                    Bxy��  �          @�ff@/\)��\@�B�u�C���@/\)���@�Q�BD��C��\                                    Bxy�>  �          @���@&ff�Q�@�z�B��\C�T{@&ff���R@���B<�C�t{                                    Bxy�#�  �          @���@-p��   @�\)Bx�RC��)@-p���  @�\)B/��C�7
                                    Bxy�2�  �          @���@,(��{@�  ByC��3@,(���\)@�Q�B0�HC�0�                                    Bxy�A0  �          @��@'��!G�@��Bz�\C�{@'�����@�  B0��C�                                    Bxy�O�  �          @�p�@%�4z�@�p�Bt  C�H�@%����@�G�B'�
C��                                    Bxy�^|  �          @�z�@'��2�\@���Bt
=C��
@'����@���B((�C�B�                                    Bxy�m"  �          @�33@#33�4z�@ۅBs��C�
=@#33��Q�@�\)B'33C��                                    Bxy�{�  �          @��@   �>�R@ڏ\Bqz�C�  @   ����@�z�B#Q�C�j=                                    Bxy��n  �          @��H@"�\�<(�@��Bq33C�c�@"�\���@�(�B#ffC��=                                    Bxy��  �          @��@(Q��1�@ۅBsz�C��\@(Q���\)@��B'(�C�Q�                                    Bxy���  �          @�z�@'
=�*=q@�ffBwG�C�E@'
=���@��B+\)C�n                                    Bxy��`  �          @��@��#33@�\B~(�C��R@����@���B1
=C���                                    Bxy��  T          @���@#33�#33@��B{G�C���@#33��33@��RB/�C�K�                                    Bxy�Ӭ  �          @�z�@���(�@��HB�\C���@������@��B3�C��q                                    Bxy��R  �          @�(�@{�{@��B~�RC���@{��G�@���B2
=C�                                    Bxy���  �          @�(�@ ���=q@��B33C��@ ����  @�G�B3{C�Q�                                    Bxy���  �          @�=q@3�
�3�
@�
=Bm�C�}q@3�
���@�=qB!�HC��                                    Bxy�D  �          @�z�@"�\�%�@�\)Bv��C�L�@"�\����@��B)��C�j=                                    Bxy��  �          @���@!G��!G�@أ�Bx�
C�z�@!G���  @��RB+�C�b�                                    Bxy�+�  �          @�{@"�\�(�@��HBz�HC��@"�\��ff@��B.�C��{                                    Bxy�:6  �          @�{@+��'
=@�
=Bs��C��H@+����@�(�B'ffC���                                    Bxy�H�  �          @���@-p��!�@�{Bt  C�t{@-p����@�(�B(z�C�AH                                    Bxy�W�  �          @�(�@<(��!G�@ҏ\Bn�C���@<(���@���B$�C�g�                                    Bxy�f(  �          @��
@&ff�z�@ۅB�8RC��q@&ff��z�@��RB7
=C���                                    Bxy�t�  �          @���@*�H�z�@�G�By��C��@*�H���H@���B/(�C�q�                                    Bxy��t  �          @�p�@/\)���@�\)Bup�C��@/\)��ff@�{B*(�C�|)                                    Bxy��  
�          @��@-p���@�Q�Bw�HC�h�@-p���(�@�  B,��C���                                    Bxy���  �          @��R@(Q��G�@�(�B|�RC���@(Q����H@�z�B1\)C�Ff                                    Bxy��f  �          @�\)@,(����@��Bw��C��@,(���\)@�  B+p�C�4{                                    Bxy��  �          @�\)@���R@�  B���C��q@����@�Q�B5��C�Ff                                    Bxy�̲  T          @�
=@   �33@�B�C��@   ����@��B2  C�|)                                    Bxy��X  �          @�  @%��Q�@��B|G�C��@%���
=@��B.�C���                                    Bxy���  �          @�\)@%�=q@��
B{  C��@%��\)@��B-z�C��q                                    Bxy���  �          @��@ ���\)@޸RB�p�C�{@ ����(�@��RB3z�C���                                    Bxy�J  �          @���@"�\��H@�B|�\C�:�@"�\����@�33B.(�C�n                                    Bxy��  �          @��@%�!G�@��HBx�\C��H@%���H@�
=B)��C�~�                                    Bxy�$�  �          @�  @���B�\@�
=Bp=qC��@������@���B��C���                                    Bxy�3<  �          @���@�H�P��@��
Bi��C�K�@�H��@�
=B�C��=                                    Bxy�A�  �          @���@�
�e�@�  Bc{C�k�@�
��p�@�
=Bz�C��f                                    Bxy�P�  �          @�G�@Q��e�@ϮBa��C�Ǯ@Q���p�@�ffB�C��\                                    Bxy�_.  �          @�  @�
�a�@�Q�Bd33C�� @�
��z�@��Bz�C���                                    Bxy�m�  �          @�Q�@
=�G�@�ffBn�HC���@
=��33@��\B�HC�g�                                    Bxy�|z  �          @�\)@
=�Fff@�p�Bn�C��@
=��=q@���BC�~�                                    Bxy��   �          @�\)@��\(�@�Q�Be33C�G�@���=q@�Q�B�
C�
                                    Bxy���  �          @��@33�\��@�{Be{C��R@33����@�ffB
=C��\                                    Bxy��l  �          @��@ff�P  @У�Bi�\C��
@ff����@��HB�\C�O\                                    Bxy��  �          @�{?�(��Q�@�B���C�aH?�(���(�@��HB:��C��q                                    Bxy�Ÿ  �          @�ff?���  @��
B�B�C��
?�����@���B8z�C�P�                                    Bxy��^  �          @�ff@
=q�4z�@ڏ\Bz(�C�� @
=q��z�@�G�B#�
C��=                                    Bxy��  �          @�\)@ff�1�@�z�B|�\C���@ff��z�@��B%�C��f                                    Bxy��  �          @�Q�@G��,��@��B{�HC�\@G����\@��B&�HC���                                    Bxy� P  �          @���@��)��@��B{�RC��f@���G�@�B'Q�C��                                    Bxy��  �          @�  @�\�+�@���B{��C�@ @�\��=q@���B&�\C���                                    Bxy��  �          @�\)@{�)��@��B}C��@{��G�@�p�B(�C�b�                                    Bxy�,B  �          @�
=@���%@��B~Q�C���@����  @�ffB)(�C���                                    Bxy�:�  �          @�{@���Q�@޸RB�p�C��q@�����\@�=qB.�HC���                                    Bxy�I�  �          @��R@(����@߮B�(�C�1�@(����@��HB/Q�C��R                                    Bxy�X4  �          @�@
�H���@�
=B�L�C��@
�H��33@�=qB/Q�C��f                                    Bxy�f�  �          @�p�@�\��@޸RB�.C���@�\��  @�33B0�HC�K�                                    Bxy�u�  �          @���@Q���@޸RB�#�C�ff@Q���=q@�{B533C�                                      Bxy��&  �          @��@33�z�@�B��C�Q�@33����@���B/�C�B�                                    Bxy���  �          @���@	���%�@ۅB��C��@	�����@�(�B(�C�(�                                    Bxy��r  �          @���@����@߮B�.C�Q�@����R@���B3�C��)                                    Bxy��  �          @�@�#33@��HB��=C���@���R@��
B)\)C��                                    Bxy���  �          @�33?�(��*�H@���B��{C�Z�?�(���33@��B)=qC��                                    Bxy��d  �          @��H@G��<(�@�Bw��C�=q@G���  @���B33C�{                                    Bxy��
  �          @�33@
=�2�\@�\)Bz  C���@
=��z�@���B �C���                                    Bxy��  �          @�(�?����0��@��
B���C��
?�����p�@�G�B%�\C�Ff                                    Bxy��V  �          @�?�\�S�
@ӅBq�C��?�\��=q@�=qB�C�u�                                    Bxz �  �          @�?��J�H@�Bup�C��q?���
=@�B��C��f                                    Bxz �  �          @�33?�33�G�@�\)Byp�C��3?�33��ff@�  BC�3                                    Bxz %H  
�          @��H?�(��>�R@�{Bw�\C�?�(����@���BC��f                                    Bxz 3�  �          @��H@	���=p�@���Bt�HC��)@	������@��B��C��q                                    Bxz B�  �          @�=q@��C�
@�33BrQ�C�XR@���33@�z�B33C�\)                                    Bxz Q:  �          @�\@��Fff@��Bo�C���@����
@��HB�C��                                     Bxz _�  �          @�=q@33�Dz�@�33Bs=qC��)@33���@�z�B=qC�                                    Bxz n�  �          @��@��B�\@ҏ\Br\)C�s3@����H@�(�B��C�h�                                    Bxz },  �          @��@\)�<��@��HBrC���@\)��  @�p�B�C�\                                    Bxz ��  �          @�G�@
�H�A�@љ�Bqp�C�˅@
�H��=q@�33B=qC��                                    Bxz �x  �          @��@��B�\@��BrC�.@����\@��HB��C�5�                                    Bxz �  �          @�ff@��:�H@У�Bt=qC���@����R@��B�HC��)                                    Bxz ��  �          @�
=?����>�R@ӅBx�C���?������@��B�C��                                    Bxz �j  �          @�?�(��L��@�\)Bo�C��3?�(���ff@�ffB�C��=                                    Bxz �  �          @�
=@�W�@��HBg��C��@��G�@��B
=qC��                                    Bxz �  �          @�\)@
=�`  @�Q�Bc�C���@
=��(�@��BffC��                                     Bxz �\  �          @�ff@33�c�
@ƸRBb�C�H@33���@�G�BQ�C��{                                    Bxz  �          @�ff@��g�@���BX�\C��@�����@uA���C�/\                                    Bxz�  �          @�\)@*�H�xQ�@���BK��C�K�@*�H����@aG�A�RC��R                                    BxzN  T          @�R@=q�n{@�  BV�C�t{@=q��
=@p��A�=qC���                                    Bxz,�  T          @�
=@#�
�n�R@�{BR��C�:�@#�
���R@l��A��C��                                     Bxz;�  �          @�R@'��z=q@�G�BK�RC��f@'����@_\)A���C��\                                    BxzJ@  �          @�
=@5��Q�@�(�BC�\C���@5��=q@R�\A�G�C��f                                    BxzX�  �          @�R@6ff�{�@��BE�C��{@6ff����@VffAծC��f                                    Bxzg�  �          @�ff@*�H�|(�@�\)BI�C��@*�H���@Z=qA�C��f                                    Bxzv2  �          @�@{�q�@���BR(�C���@{��  @g�A�\)C�,�                                    Bxz��  �          @�p�@p��w
=@��BT�C��=@p��\@fffA�RC��                                    Bxz�~  �          @�@�
�j=q@��
B^=qC��\@�
���@w�A��C�|)                                    Bxz�$  �          @�(�@
=q�u@�BU�C��=@
=q��=q@g
=A�ffC���                                    Bxz��  �          @�p�@	������@��HBOz�C���@	���ƸR@\(�AܸRC���                                    Bxz�p  �          @�?�Q���z�@�Q�BNQ�C��{?�Q���Q�@U�A��C��q                                    Bxz�  �          @�33?��H���
@���BN�C��R?��H��  @UA��C���                                    Bxzܼ  �          @�\@��{@�z�BI33C�:�@��Q�@L(�A�{C�5�                                    Bxz�b  �          @陚@
=��p�@��BH�RC�k�@
=��\)@J=qA�G�C�Z�                                    Bxz�  �          @陚@   ��
=@�B?�HC��@   ��p�@7�A���C���                                    Bxz�  �          @���@�\���@�33BI��C�{@�\��\)@J=qA�C��                                    BxzT  �          @�Q�?�����z�@�BB
=C���?�����33@9��A�Q�C���                                    Bxz%�  �          @�?����@���B1�HC��R?����=q@�A���C��
                                    Bxz4�  �          @�Q�@�\��@��B`=qC��H@����@s�
A��HC���                                    BxzCF  �          @�\)@  �z�@θRB}��C��@  ��
=@�\)B#�C�+�                                    BxzQ�  �          @�
=@�
�)��@�z�By�C��@�
���@���B  C��)                                    Bxz`�  �          @�?��*=q@�\)B}�
C�W
?���G�@��HB\)C��R                                    Bxzo8  �          @�{@	���G�@��HBhz�C�N@	������@�G�B��C���                                    Bxz}�  �          @�@���E�@��HBh��C���@�����@���B	��C��                                    Bxz��  �          @���@�����
@�=qBC��C��q@����=q@8��A�p�C���                                    Bxz�*  �          @���?�
=���@��B4\)C�@ ?�
=��{@=qA�C�W
                                    Bxz��  T          @�?��R��{@��RB1{C�|)?��R��p�@�
A��C��
                                    Bxz�v  T          @�p�?����Tz�@���Bd��C���?�����(�@p��B��C��                                    Bxz�  �          @�{?�p����@�z�B���C�~�?�p���p�@�ffB.(�C��3                                    Bxz��  �          @�{?�ff���H@�  B��=C�C�?�ff���\@��B;  C���                                    Bxz�h  �          @�ff?�=q�\@�G�B�L�C�%?�=q��{@��B@�C�8R                                    Bxz�  �          @�  ?�z�˅@ٙ�B���C��?�z���Q�@��\B=C�y�                                    Bxz�  �          @�
=?�׿�{@ָRB���C��{?�����R@�(�B5ffC��                                    BxzZ  �          @�R?�=q��
=@�{B��C���?�=q��Q�@��\B3�C��                                    Bxz   �          @�R?�33��R@��HB�#�C�7
?�33��\)@�33B)(�C�j=                                    Bxz-�  �          @�@Q��(��@�z�Bx  C�xR@Q�����@�\)B  C��                                    Bxz<L  �          @�R@���,��@ʏ\BuC�0�@����G�@���BffC�                                      BxzJ�  �          @�ff?�p��	��@љ�B��fC�Ff?�p����@��HB)��C��
                                    BxzY�  �          @�?�׿��H@ڏ\B��\C��?����p�@���BAQ�C�~�                                    Bxzh>  
�          @�R?���,(�@��B{��C�l�?�����\@�
=BG�C���                                    Bxzv�  �          @�
=?��8��@ʏ\Bu  C��f?���\)@���B�C���                                    Bxz��  �          @�ff?�  �+�@�{B~=qC���?�  ���H@��B�\C�R                                    Bxz�0  �          @�z�?��33@ϮB�8RC�T{?�����@��RB%ffC��                                    Bxz��  �          @��@   �QG�@���Bfz�C��@   ��@w�B�RC��R                                    Bxz�|  �          @�@	���.{@�G�Bt�C�0�@	�����@��HB�C��                                    Bxz�"  �          @�@
�H�(��@���Bu�
C���@
�H���@��B(�C�:�                                    Bxz��  �          @�{@��\)@�B}�HC���@���@��BffC��3                                    Bxz�n  �          @�ff@\)�L(�@�G�BdC��{@\)���
@x��B  C��                                    Bxz�  �          @�p�@#33�Z�H@���BV  C�U�@#33��ff@a�A�  C��                                    Bxz��  �          @�z�@'��u�@�BE��C�7
@'�����@B�\A�C���                                    Bxz	`  �          @�(�@2�\�p��@��
BC�C�J=@2�\��=q@AG�Aȏ\C��                                    Bxz  �          @��
@7
=�x��@�\)B=
=C�  @7
=���
@5�A�33C��
                                    Bxz&�  �          @�(�@9�����
@���B3��C��\@9�����@#33A��C��                                    Bxz5R  �          @�z�@���x��@��BH�HC��@�����@C33A�z�C��q                                    BxzC�  �          @�=q@Q��hQ�@�BU�
C�=q@Q���33@UA��
C�f                                    BxzR�  �          @�@ ���c33@��B[C��@ �����H@_\)A��
C�~�                                    BxzaD  �          @�z�?�  �N�R@��
Bl{C�33?�  ���R@z=qB�C���                                    Bxzo�  �          @��
?�  �G
=@��Bo��C��R?�  ���
@�  B��C���                                    Bxz~�  �          @�G�?���H��@�=qBp33C��?�����@y��Bz�C��\                                    Bxz�6  �          @�(�?����,(�@��HB~\)C��3?�����
=@�(�B��C�ff                                    Bxz��  �          @�\)?���Z=q@��Bd�\C�ٚ?����  @eA���C���                                    Bxz��  �          @�\)?�G��[�@�(�Bf{C�޸?�G�����@eA�ffC�O\                                    Bxz�(  T          @��H?����`  @��RBdz�C�O\?�����(�@hQ�A��C��H                                    Bxz��  
�          @�  ?����mp�@�BZffC��?�����{@Q�A�RC��f                                    Bxz�t  �          @���?У��xQ�@�(�BT�RC�h�?У���=q@I��A�
=C���                                    Bxz�  �          @ᙚ?�33��p�@�p�BIffC�˅?�33��\)@4z�A�\)C�l�                                    Bxz��  �          @��?ٙ����@�
=BLC�b�?ٙ�����@:�HA���C��)                                    Bxzf  �          @ᙚ?���h��@��
Ba�RC�k�?����
=@^{A�{C���                                    Bxz  �          @�G�?�G��tz�@��RBY
=C�˅?�G����@O\)AۅC�H                                    Bxz�  �          @���@
=q��
=@�B>33C���@
=q���@%A���C��                                    Bxz.X  �          @�Q�@{��
=@�p�B2ffC�.@{��Q�@�RA���C��q                                    Bxz<�  �          @�ff@   ��p�@���B!�
C�R@   �Ǯ?��
AmG�C��=                                    BxzK�  �          @�
=@���Q�@�(�B2
=C�` @�����@�A��C�(�                                    BxzZJ  �          @޸R@ �����\@��HB0G�C���@ ����=q@ffA���C���                                    Bxzh�  �          @�ff?����@��
B1��C���?����p�@�A�\)C�+�                                    Bxzw�  �          @�{?�z���p�@�
=B,G�C�!H?�z���33?���A���C�aH                                    Bxz�<  T          @��?�p�����@��B0�\C�?�p�����@A��\C��q                                    Bxz��  �          @�z�?��H����@�B+Q�C�l�?��H��=q?�A�\)C��                                     Bxz��  �          @���?���33@�G�B0�C��?���=q@�\A��C�                                      Bxz�.  �          @��?�Q����H@�(�B3��C��?�Q��˅@
=A��\C�q�                                    Bxz��  �          @�?�G�����@��B033C�` ?�G���(�@�A�(�C��R                                    Bxz�z  �          @޸R?��H��ff@��HB0G�C��)?��H��@G�A�\)C�n                                    Bxz�   �          @޸R?��H���\@��RB5�C�5�?��H��z�@(�A�Q�C�y�                                    Bxz��  �          @޸R?�(���G�@��B7G�C�b�?�(����
@�RA���C��\                                    Bxz�l  �          @޸R?�G���
=@���B9Q�C��)?�G��ʏ\@�\A�(�C���                                    Bxz
  �          @�{?���(�@�=qB;�C�33?��ȣ�@�A���C�
=                                    Bxz�  �          @�{?�����@��
B>=qC���?���
=@��A�G�C�@                                     Bxz'^  �          @�{?�z����@���B:�HC�ٚ?�z��Ǯ@
=A�
=C��                                    Bxz6  �          @�@
=���\@�\)B7C��
@
=��@33A�p�C�g�                                    BxzD�  �          @�p�@�����@���B/�RC�` @���Q�@33A�{C�33                                    BxzSP  �          @��@�R��G�@�ffB+�C��@�R��\)?��HA��C���                                    Bxza�  �          @�p�@������@��\B1(�C���@����@Q�A��C��                                    Bxzp�  �          @ۅ?������@�=qB&�
C�4{?������H?޸RAk
=C��                                    BxzB  �          @�p�?У���\)@���B\)C�~�?У��\?��\A6{C�~�                                    Bxz��  �          @�=q?����{@l(�B��C���?����33?^�R@�(�C�e                                    Bxz��  �          @��?��\��@J=qA�(�C�4{?��\���H>���@u�C��                                    Bxz�4  �          @���?��\����@J=qB��C�ff?��\��\)>��@��C�=q                                    Bxz��  �          @�\)?��R��G�@P��B
��C��q?��R����?.{@�=qC�G�                                    BxzȀ  �          @�(�?����z�@W�B�HC�˅?����{?8Q�@�=qC�h�                                    Bxz�&  �          @�=q?�
=��ff@N{BG�C��?�
=��p�?\)@�  C��f                                    Bxz��  �          @���?����@EBz�C��3?���z�>�(�@���C�޸                                    Bxz�r  �          @��?��R���H@:=qA��C��{?��R���=�Q�?^�RC���                                    Bxz  �          @�p�?��H��
=@C�
A�  C���?��H����>.{?�ffC��                                    Bxz�  �          @��R?�33����@0��A��C�K�?�33���H�u�(�C���                                    Bxz d  �          @��H?��
��\)@1G�A�ffC�/\?��
��ff<��
>L��C�O\                                    Bxz/
  �          @��H?������@Mp�B33C�"�?�����ff?�@�z�C��R                                    Bxz=�  �          @��?z�H��{@@  B G�C��f?z�H����>�Q�@j�HC�                                      BxzLV  �          @��\?������@?\)B��C���?�����{>��@�\)C��                                    BxzZ�  �          @��?������
@8��A�ffC��=?�����{>�z�@@  C���                                    Bxzi�  �          @���?����Q�@-p�A�z�C��)?����p��.{�У�C�8R                                    BxzxH  �          @˅?�\)����@,(�Aȏ\C��H?�\)�Ǯ�\�Z�HC��                                    Bxz��  �          @��H?�33���
@)��A�z�C��\?�33��ff�����j=qC�Ff                                    Bxz��  �          @�p�?�{��=q@S33A�  C�.?�{��  >���@.�RC�%                                    Bxz�:  �          @��?��R���@�G�B\)C��H?��R��33?�p�A1G�C��H                                    Bxz��  �          @�{?Ǯ��G�@�Q�B{C��?Ǯ��(�?�
=A(��C�!H                                    Bxz��  �          @�(�?��
���\@���B�HC��)?��
��
=?��A>=qC�K�                                    Bxz�,  �          @�G�@���p�@�{B'�C�|)@���?��Aq�C�<)                                    Bxz��  �          @���?�(���Q�@��B&33C�o\?�(���  ?ǮAg
=C�|)                                    Bxz�x  �          @�33?�
=��  @�G�B6\)C��)?�
=��ff@G�A�  C�W
                                    Bxz�  �          @�Q�@��n�R@�z�BCz�C��\@���z�@p�A�C�q                                    Bxz
�  �          @ҏ\@�\�r�\@�
=BDp�C�.@�\��\)@   A��\C���                                    Bxzj  �          @�p�?�������@��B=��C���?�����@�
A�ffC�)                                    Bxz(  �          @�p�@���h��@�33BG�C��@�����@+�A�C��H                                    Bxz6�  �          @��@��U@�G�BRffC�%@���  @?\)A�G�C�AH                                    BxzE\  �          @�z�@��g
=@��\BG��C�(�@����
@*�HA�  C�
=                                    BxzT  �          @�(�@{�g�@�=qBHG�C���@{��z�@*=qA��C���                                    Bxzb�  �          @��
@
=�u@�ffBA��C�` @
=����@(�A�33C��                                    BxzqN  �          @Ӆ@���Q�@���B9(�C�B�@����H@p�A�  C�>�                                    Bxz�  I          @��
@Q���33@��B6�C���@Q���z�@Q�A�{C��\                                    Bxz��  �          @Ӆ@ ����{@��RB5��C��3@ �����R@�
A��C�S3                                    Bxz�@  �          @�33@   ��\)@���B3�C���@   ��
=?�p�A��C�@                                     Bxz��  
�          @�=q@���33@�{B6\)C�|)@����
@�A�  C��f                                    Bxz��  
�          @�=q@�
��Q�@���B:�
C��
@�
���\@��A���C���                                    Bxz�2  
Z          @ҏ\@{���
@�(�B2�C�&f@{���@G�A���C�\)                                    Bxz��  
�          @��H@
=q���R@��\B0(�C���@
=q��p�?�A��C�f                                    Bxz�~  �          @���@(���p��@���B8�HC���@(����(�@A��\C���                                    Bxz�$  
�          @��
@#�
�\(�@�G�BFG�C�XR@#�
��
=@,��A��\C��                                    Bxz	�  -          @��
@(Q��u�@��RB5�C�5�@(Q�����@�RA��C�t{                                    Bxz	p  
�          @Ӆ@   �xQ�@��B7(�C�e@   ��ff@{A�G�C���                                    Bxz	!  
Z          @ҏ\@�R�g
=@��B@�
C�@ @�R���@   A�ffC��                                    Bxz	/�  T          @Ӆ@z����H@��B7��C�o\@z���z�@
=A�p�C���                                    Bxz	>b  
�          @��H@�dz�@�Q�BFp�C��q@��=q@'
=A���C�`                                     Bxz	M  �          @��@  �xQ�@���B;33C��@  ��\)@  A���C��
                                    Bxz	[�  _          @�=q@�����@�ffB6��C�/\@���=q@ffA��C�AH                                    Bxz	jT  
�          @Ӆ@z��\)@��\B<{C��)@z���33@\)A�z�C���                                    Bxz	x�  �          @��
@
�H�k�@���BG�C�<)@
�H��{@%�A��
C�j=                                    Bxz	��  �          @�=q@��^�R@�z�BM�\C��@����@0��A�ffC���                                    Bxz	�F  I          @���@��QG�@�p�BQz�C�z�@���z�@8��A�{C�z�                                    Bxz	��  "          @У�@   �L��@��\BM�C�f@   ��G�@6ffA�p�C��=                                    Bxz	��  
�          @У�@   �L(�@�33BM�C�f@   ��G�@7
=A�{C��                                    Bxz	�8            @�G�@'��O\)@�G�BI�C���@'�����@2�\A�C�*=                                    Bxz	��  �          @�G�@7
=�W�@���B=p�C�1�@7
=����@ ��A�33C�33                                    Bxz	߄  T          @�
=@?\)�?\)@���BD�C�y�@?\)����@1�A��C�^�                                    Bxz	�*  �          @Ϯ@=p��<��@�ffBGQ�C���@=p���Q�@6ffAЏ\C�J=                                    Bxz	��  
�          @�ff@333�&ff@��BV  C��{@333���@L��A�p�C�)                                    Bxz
v  _          @�{@-p��S33@��BA��C��
@-p���  @#33A���C���                                    Bxz
  
�          @��@&ff�\��@�
=B>=qC��H@&ff���\@��A�(�C���                                    Bxz
(�  
�          @ʏ\@#33�W
=@��RB@�RC�� @#33��  @(�A�33C��
                                    Bxz
7h  �          @��
@$z��%@�ffB[�HC�l�@$z���=q@O\)A�p�C�
=                                    Bxz
F  "          @�(�@"�\� ��@��Bm�C���@"�\���R@qG�B�C��{                                    Bxz
T�  �          @θR@���(��@��B`��C�q�@����ff@W
=A�ffC�+�                                    Bxz
cZ  "          @�
=@A���@�ffBY=qC�*=@A����R@\(�B��C�R                                    Bxz
r   �          @�\)@<�Ϳ�(�@�  Bh�C��=@<����
=@{�BffC��                                     Bxz
��  "          @�G�@.{����@���Bo=qC�� @.{���@\)B�C��{                                    Bxz
�L  T          @љ�@����@�{BqG�C��@�����R@w
=B�C��H                                    Bxz
��  
�          @љ�@�R���@�=qBi  C�@�R���\@i��B�\C��
                                    Bxz
��  
�          @љ�@#�
�ff@�G�Bg�RC��{@#�
��G�@i��B�\C��                                    Bxz
�>  �          @ҏ\@8����
@�G�Be�C�Ff@8����G�@r�\BC�@                                     Bxz
��  
(          @љ�@.�R�z�@�=qBi�\C�q�@.�R���@s�
B�C�y�                                    Bxz
؊  T          @�=q@>�R��\@��Bh33C���@>�R����@|��B=qC�b�                                    Bxz
�0  �          @��
@=p���
@���Bd  C���@=p���G�@r�\B�C��                                    Bxz
��  �          @Ӆ@:�H�z�@���Bd�RC�Z�@:�H����@r�\B�C�Y�                                    Bxz|  "          @Ӆ@333��\@���Bc  C�ff@333��\)@j=qB�C�W
                                    Bxz"  T          @��
@9���  @�  BaG�C�R@9����@j=qBffC��q                                    Bxz!�  
�          @�=q@7
=�(�@�(�B\
=C��H@7
=��G�@\��A��
C�l�                                    Bxz0n  T          @�G�@.{�=q@�{Ba  C�4{@.{����@aG�B=qC��                                    Bxz?  T          @�G�@5�Q�@�z�B]��C��
@5��  @_\)B  C�l�                                    BxzM�  T          @�=q@5��
=@�{B_�\C�
@5���  @c33B�C�c�                                    Bxz\`  �          @љ�@.�R�#33@�(�B\�HC�y�@.�R��z�@Y��A�{C��
                                    Bxzk  �          @�=q@*=q�#�
@�B_Q�C�@*=q��p�@\(�A���C�*=                                    Bxzy�  "          @У�@#33�(��@�z�B^�HC�  @#33��
=@W
=A�{C��{                                    Bxz�R  �          @��@/\)�'�@�33BZC�#�@/\)��@UA���C���                                    Bxz��  T          @�=q@*=q�#33@�p�B_
=C�R@*=q���@[�A�ffC�9�                                    Bxz��  
�          @�=q@#33�@  @�  BTz�C�'�@#33��
=@C�
A�=qC��                                    Bxz�D  
�          @ҏ\@\)�Q�@�(�BM{C���@\)����@4z�A���C�]q                                    Bxz��  
�          @�33@   �P��@��BM�C�Ǯ@   ��z�@6ffA��C�s3                                    Bxzѐ  
�          @�=q@�R�J=q@�{BQ  C�\@�R���\@;�A�C�|)                                    Bxz�6  �          @�  @{��G�@�ffB,��C���@{��
=?�A��C���                                    Bxz��  �          @�Q�@p���  @�Q�B/�C��\@p����R?�A�\)C��f                                    Bxz��  "          @�G�@=q�u�@�
=B8�C�
@=q��p�@(�A�Q�C��H                                    Bxz(  
�          @�=q@   �i��@��B>�HC�>�@   ���\@��A�\)C�3                                    Bxz�  �          @ҏ\@$z��<��@���BU�C���@$z���{@FffA�C�0�                                    Bxz)t  
�          @���@#�
�aG�@��B@C�H@#�
��
=@p�A�p�C��                                    Bxz8            @љ�@!��`��@�z�BA�C��f@!���\)@\)A�33C�ff                                    BxzF�  �          @љ�@'��J�H@�33BLffC���@'���G�@5A�G�C�&f                                    BxzUf  
�          @ҏ\@#33�R�\@��BJ=qC��=@#33���
@0  A�Q�C��{                                    Bxzd  "          @љ�@�R�@��@�Q�BU��C��@�R��\)@C�
Aޏ\C��{                                    Bxzr�  �          @��@�R�1�@���B]��C��)@�R��33@R�\A�G�C��{                                    Bxz�X  
�          @љ�@$z��(��@�p�B_(�C��@$z����@XQ�A�z�C��                                     Bxz��  �          @љ�@!��2�\@��
B[�C��@!����H@P��A�z�C�8R                                    Bxz��  �          @ҏ\@!G��%�@�  Bb�C�%@!G���\)@^�RA�p�C�j=                                    Bxz�J  "          @���@�333@��Bc  C��
@��ff@[�A�{C��                                    Bxz��  �          @�(�@�R�HQ�@���BZ\)C���@�R���@G�A�RC�/\                                    Bxzʖ  T          @�p�@�R�W
=@���BS  C�˅@�R���@:�HA��C��                                    Bxz�<  
�          @Ӆ@&ff�.�R@�33B[ffC���@&ff��G�@Q�A�Q�C���                                    Bxz��  
�          @��H@0  �   @�B^z�C��=@0  ���
@]p�A�  C��q                                    Bxz��  �          @Ӆ@'
=�9��@��HBWG�C�H@'
=��@K�A��
C�c�                                    Bxz.  "          @�z�@$z��A�@�=qBU�C�%@$z�����@FffA޸RC��
                                    Bxz�  T          @�(�@   �J=q@�Q�BR�C�(�@   ���
@>�RA�C�}q                                    Bxz"z  
Z          @Ӆ@ ���Dz�@�G�BT��C��H@ ������@C33A�{C��=                                    Bxz1   
�          @�(�@=q�@  @���BZ33C�\)@=q����@K�A噚C�7
                                    Bxz?�  {          @�z�@Q��O\)@�G�BS  C�"�@Q���ff@>{A�  C�˅                                    BxzNl  
�          @Ӆ@���]p�@��HBI�
C�Q�@����G�@,(�A�{C���                                    Bxz]  �          @Ӆ@\)�Vff@�(�BK�\C�O\@\)���R@1G�A�z�C�>�                                    Bxzk�  	          @�33@#�
�Mp�@�p�BN(�C�B�@#�
���@8Q�AΣ�C���                                    Bxzz^  
Z          @��H@%��G�@��RBP�C�@%�����@<��A���C��3                                    Bxz�  {          @��H@$z��L(�@��BN{C�o\@$z����H@8Q�A��C��)                                    Bxz��  �          @Ӆ@-p��5@��BV33C��)@-p����@L(�A���C���                                    Bxz�P  �          @�(�@1��5@���BTC�0�@1���33@K�A�C�E                                    Bxz��  �          @�(�@-p��/\)@�z�BY�RC�Q�@-p���=q@S33A��C��                                    BxzÜ  �          @��H@$z��2�\@�(�B[G�C�Z�@$z���33@QG�A�ffC�b�                                    Bxz�B  �          @�=q@!G��?\)@�G�BV  C��@!G���\)@EA�  C�ٚ                                    Bxz��  �          @��
@)���8��@�=qBVffC�E@)�����@J�HA�33C���                                    Bxz�  �          @��
@(���*=q@��RB^=qC�k�@(������@Z=qA�C���                                    Bxz�4  �          @��
@+��0��@�(�BY�\C��@+���=q@Q�A�{C��                                    Bxz�  �          @Ӆ@0  �'
=@��B[��C�>�@0  ���R@X��A���C�~�                                    Bxz�  �          @�(�@5�*=q@��
BX(�C�c�@5���@Tz�A�z�C�Ф                                    Bxz*&  �          @�z�@1��8��@���BR��C��\@1���z�@HQ�A�\)C�5�                                    Bxz8�  �          @�z�@7
=�;�@�
=BOffC�#�@7
=����@C�
Aۙ�C��=                                    BxzGr  �          @�@3�
�@  @�Q�BP�C��\@3�
��
=@Dz�AڸRC�                                      BxzV  �          @�z�@+��E@�\)BOC�xR@+���G�@?\)A�  C�s3                                    Bxzd�  �          @Ӆ@1G��I��@�33BJ\)C���@1G�����@6ffA��HC��q                                    Bxzsd  �          @�33@0���P��@���BF�C�)@0�����\@.�RA��
C��=                                    Bxz�
  �          @�33@2�\�S33@�\)BD(�C�#�@2�\���H@*�HA�G�C��\                                    Bxz��  �          @ҏ\@5�N�R@�\)BD��C���@5��G�@-p�A�Q�C�)                                    Bxz�V  �          @Ӆ@O\)�5�@�Q�BE\)C�h�@O\)��ff@;�A���C��
                                    Bxz��  �          @�33@O\)�Fff@��\B<p�C�#�@O\)��33@(��A���C�<)                                    Bxz��  �          @�(�@Mp��@��@��RBA��C�o\@Mp����\@333A�{C�#�                                    Bxz�H  �          @�@J=q�Tz�@��B:��C���@J=q����@#�
A�  C�ff                                    Bxz��  �          @�{@J=q�U@��
B:��C���@J=q��=q@#�
A�C�]q                                    Bxz�  �          @�@P  �[�@�
=B3��C��3@P  ���\@��A�G�C���                                    Bxz�:  
�          @�p�@N{�W
=@���B6��C���@N{��G�@p�A�p�C��=                                    Bxz�  �          @�p�@R�\�R�\@���B7Q�C��H@R�\��\)@ ��A���C��                                    Bxz�  �          @�{@O\)�Q�@��HB9��C�N@O\)��Q�@$z�A�ffC�Ф                                    Bxz#,  �          @�ff@I���\(�@���B7(�C�=q@I����(�@��A��C�1�                                    Bxz1�  �          @���@HQ��S33@�33B;G�C��R@HQ�����@#�
A�33C�S3                                    Bxz@x  �          @�z�@J�H�P  @�33B;��C�(�@J�H��\)@%A���C��q                                    BxzO  �          @���@K��S�
@��B9=qC���@K���Q�@!G�A�Q�C���                                    Bxz]�  �          @�z�@N{�=p�@��BC(�C���@N{����@7
=A�=qC�E                                    Bxzlj  �          @�(�@J=q�9��@��BF�
C�� @J=q����@<��A��C�{                                    Bxz{  �          @�z�@L���7
=@��BF��C��@L�����@>{Aԣ�C�S3                                    Bxz��  �          @��@P���3�
@��\BF�HC��)@P����
=@@��A���C��                                    Bxz�\  �          @���@N�R�1�@��HBH33C���@N�R��ff@B�\AمC���                                    Bxz�  �          @�(�@J=q�Fff@�{B@��C���@J=q����@0  A�C���                                    Bxz��  �          @���@Q��L��@��B9��C��H@Q���p�@%A�\)C�7
                                    Bxz�N  �          @�p�@Mp��e@��B.�HC��@Mp����@p�A��C�Y�                                    Bxz��  �          @�{@N�R�|(�@��HB!�C��\@N�R��=q?�A{
=C��                                    Bxz�  �          @�ff@N{�|��@��HB!��C���@N{���\?�Az�HC��q                                    Bxz�@  �          @��@J�H���H@�ffBffC�� @J�H��(�?�\)A`��C��=                                    Bxz��  "          @�p�@L������@��RB�C�!H@L�����?�33Ad��C���                                    Bxz�  �          @��@Mp�����@�
=B�C�L�@Mp����\?�Ah��C��                                    Bxz2  �          @�z�@J�H�~�R@���B =qC�Ff@J�H��=q?޸RAr�RC��\                                    Bxz*�  �          @�z�@HQ��x��@��
B$��C�k�@HQ�����?�{A�ffC��)                                    Bxz9~  �          @��@Q��hQ�@���B*��C�\@Q���z�@�A���C���                                    BxzH$  �          @�(�@G
=�q�@�
=B)��C���@G
=��  @ ��A���C���                                    BxzV�  �          @�z�@Fff���\@�
=B�HC��R@Fff��(�?��Ae�C�e                                    Bxzep  �          @���@Dz���(�@�ffB
=C�O\@Dz���p�?�{A_�
C�5�                                    Bxzt  �          @��
@G
=��
=@�G�B\)C�33@G
=��p�?�AE��C�c�                                    Bxz��  H          @�z�@HQ����\@�B�\C��R@HQ����
?�{Aap�C��=                                    Bxz�b  �          @�33@6ff���R@{�BffC�C�@6ff��=q?��HA(��C�f                                    Bxz�  �          @ҏ\@C�
����@��HB��C�8R@C�
���
?�  AR�RC�G�                                    Bxz��  �          @�33@6ff��G�@uB�\C�3@6ff��33?���A��C�H                                    Bxz�T  T          @��
@A����
@{�B��C�Y�@A����?�  A.�\C��f                                    Bxz��  T          @�z�@G����
@�B��C��\@G�����?���A^�\C�s3                                    Bxzڠ  �          @�z�@Tz��S33@�{B4(�C���@Tz���@(�A�33C�S3                                    Bxz�F  	�          @���@K��`  @�p�B2�C�!H@K���33@�A�z�C�\)                                    Bxz��  
�          @Ӆ@<������@}p�B{C��\@<������?��\A1�C��f                                    Bxz�  
�          @Ӆ@Dz��vff@��B'\)C�G�@Dz�����?�
=A�  C���                                    Bxz8  
Z          @��
@<(��aG�@���B8�\C��=@<(���p�@=qA���C�:�                                    Bxz#�  �          @Ӆ@=p��`  @��B7�C�R@=p���(�@��A�{C�e                                    Bxz2�  
�          @��
@C�
�g
=@�(�B1�\C�!H@C�
��p�@  A���C��{                                    BxzA*  
�          @�33@E�h��@���B.��C�*=@E���@
�HA�p�C��)                                    BxzO�  
Z          @ҏ\@<���fff@���B3C���@<����p�@G�A�G�C�E                                    Bxz^v  
Z          @�Q�@B�\�+�@���BL�
C�Q�@B�\��=q@Dz�A�z�C�#�                                    Bxzm  T          @�ff@5� ��@��RBW�C�B�@5��  @R�\A��C�q�                                    Bxz{�  T          @�ff@1��5@��\BO�HC�33@1���
=@AG�A߮C���                                    Bxz�h  �          @�
=@333�E@�ffBG��C�q@333��(�@2�\A�Q�C�U�                                    Bxz�  
�          @У�@:=q�A�@��BG��C���@:=q��33@6ffA�\)C�ٚ                                    Bxz��  
�          @�=q@8Q��@��@�=qBJ\)C���@8Q����@;�Aԏ\C��{                                    Bxz�Z  
�          @���@6ff�Mp�@�p�BC��C��@6ff��
=@-p�A�  C�U�                                    